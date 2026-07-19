from __future__ import annotations

import copy

import numpy as np
import pytest

from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import entailmentbank_proof_retrieval_core_v1 as core
from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_runtime_v1 as runtime,
)
from replication_runtime.qasc_nli_v1.contract import NLIPair
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        values = tuple(texts)
        self.calls.append(values)
        matrix = np.zeros((len(values), minilm_binding.EMBEDDING_DIMENSION), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix


class _FakeNLI:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, tuple[NLIPair, ...]], ...]] = []

    def score_items(self, items):
        normalized = tuple((key, tuple(pairs)) for key, pairs in items)
        self.calls.append(normalized)
        return {
            key: tuple(core.token_f1(pair.premise, pair.hypothesis) for pair in pairs)
            for key, pairs in normalized
        }


def _item(index: int) -> core.LabelFreeItem:
    return core.LabelFreeItem(
        f"{index + 1:064x}",
        f"which private node {index}",
        f"answer {index}",
        f"private hypothesis alpha {index}",
        tuple(
            f"private fact alpha {index} node {ordinal}" for ordinal in range(25)
        ),
    )


def _tensor(index: int) -> core.ItemTensor:
    item = _item(index)
    pair = core.build_pair_token_f1(item.node_texts)
    return core.ItemTensor(
        item.item_commitment_sha256,
        tuple((index,) * core.NODE_FEATURE_COUNT for _ in range(25)),
        pair,
    )


def _view_pack(block: str) -> dict[str, object]:
    rows = []
    for ordinal in range(acquisition.BLOCK_COUNTS[block]):
        item = _item(ordinal + (10_000 if block == "M_search" else 0))
        rows.append(
            {
                "ordinal": ordinal,
                "item_commitment_sha256": item.item_commitment_sha256,
                "question": item.question,
                "answer": item.answer,
                "hypothesis": item.hypothesis,
                "node_texts": list(item.node_texts),
            }
        )
    body = {
        "schema": f"{acquisition.VERSION}_block_view",
        "block": block,
        "source_split": "dev" if block == "M_search" else "train",
        "item_count": len(rows),
        "items": rows,
        "excluded_fields": [
            "proof",
            "meta.distractors",
            "meta.intermediate_conclusions",
            "gold_leaf_IDs",
            "family",
            "source_item_ID",
        ],
    }
    return acquisition.self_hashed(body, "pack_sha256")


def _label_pack(block: str) -> dict[str, object]:
    rows = []
    family_rows = (
        ("TWO_LEAF", (0, 1)),
        ("THREE_LEAF", (0, 1, 2)),
        ("FOUR_FIVE_LEAF", (0, 1, 2, 3)),
    )
    ordinal = 0
    for family, gold in family_rows:
        for _ in range(acquisition.BLOCK_FAMILY_COUNTS[block][family]):
            rows.append(
                {
                    "ordinal": ordinal,
                    "item_commitment_sha256": _item(ordinal).item_commitment_sha256,
                    "family": family,
                    "gold_ordinals": list(gold),
                }
            )
            ordinal += 1
    body = {
        "schema": f"{acquisition.VERSION}_block_labels",
        "block": block,
        "source_split": "dev" if block == "M_search" else "train",
        "item_count": len(rows),
        "items": rows,
    }
    return acquisition.self_hashed(body, "pack_sha256")


def test_feature_builder_uses_one_minilm_call_and_exactly_50_nli_pairs_per_item() -> None:
    items = (_item(0), _item(1))
    encoder = _FakeEncoder()
    nli = _FakeNLI()
    tensors = runtime.build_item_tensors(
        items, minilm_encoder=encoder, nli_scorer=nli
    )
    assert len(tensors) == 2
    assert len(encoder.calls) == 1 and len(encoder.calls[0]) == 54
    assert len(nli.calls) == 1
    assert [len(pairs) for _key, pairs in nli.calls[0]] == [50, 50]
    assert tensors[0].node_features[0][2:4] == (
        core.INTEGER_SCALE,
        core.INTEGER_SCALE,
    )
    assert tensors[0].node_features[0][0] == core.token_f1(
        items[0].node_texts[0], items[0].hypothesis
    )
    assert tensors[0].node_features[0][1] == core.token_f1(
        items[0].node_texts[0], items[0].answer_query
    )


def test_feature_builder_rejects_duplicate_commitments_before_model_calls() -> None:
    encoder = _FakeEncoder()
    nli = _FakeNLI()
    with pytest.raises(runtime.EntailmentBankRuntimeError, match="duplicated"):
        runtime.build_item_tensors(
            (_item(0), _item(0)), minilm_encoder=encoder, nli_scorer=nli
        )
    assert encoder.calls == [] and nli.calls == []


def test_view_and_label_pack_decoders_are_exact_and_self_hashed() -> None:
    views = runtime.decode_view_pack(_view_pack("G_form"), block="G_form")
    labels = runtime.decode_label_pack(_label_pack("G_form"), block="G_form")
    assert len(views) == len(labels) == acquisition.BLOCK_COUNTS["G_form"]
    assert tuple(item.item_commitment_sha256 for item in views) == tuple(
        label.item_commitment_sha256 for label in labels
    )
    leaked = copy.deepcopy(_view_pack("G_form"))
    leaked["items"][0]["family"] = "TWO_LEAF"
    leaked.pop("pack_sha256")
    leaked = acquisition.self_hashed(leaked, "pack_sha256")
    with pytest.raises(runtime.EntailmentBankRuntimeError, match="keys"):
        runtime.decode_view_pack(leaked, block="G_form")
    with pytest.raises(runtime.EntailmentBankRuntimeError, match="invalid"):
        runtime.decode_label_pack(_label_pack("G_form"), block="F_search")


def test_tensor_pack_roundtrip_is_exact_and_tamper_fails_closed() -> None:
    tensors = tuple(
        _tensor(index) for index in range(acquisition.BLOCK_COUNTS["G_form"])
    )
    pack = runtime.tensor_pack("G_form", tensors)
    assert runtime.decode_tensor_pack(pack, block="G_form") == tensors
    tampered = copy.deepcopy(pack)
    tampered["items"][0]["node_features"][0][0] += 1
    with pytest.raises(runtime.EntailmentBankRuntimeError, match="hash"):
        runtime.decode_tensor_pack(tampered, block="G_form")


def test_entailmentbank_nli_pool_rejects_any_worker_count_other_than_two(tmp_path) -> None:
    with pytest.raises(runtime.EntailmentBankRuntimeError, match="exactly two"):
        runtime.LocalTwoWorkerNLIPool(project_root=tmp_path, workers=3)
