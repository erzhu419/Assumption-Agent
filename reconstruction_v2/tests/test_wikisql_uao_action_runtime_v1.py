from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
import stat
from pathlib import Path

import pytest

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as runtime,
)
from assumption_agent.benchmarks import wikisql_uao_policy_v1 as policy
from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as source_compiler,
)


def test_runtime_bounds_exactly_match_pre_hmac_source_eligibility() -> None:
    assert runtime.MAX_COLUMNS == source_compiler.MAX_COLUMNS
    assert (
        runtime.MAX_QUESTION_CHARACTERS
        == source_compiler.MAX_QUESTION_CHARACTERS
    )
    assert (
        runtime.MAX_HEADER_CHARACTERS
        == source_compiler.MAX_HEADER_OR_CELL_CHARACTERS
    )
    assert (
        runtime.MAX_CELL_CHARACTERS
        == source_compiler.MAX_HEADER_OR_CELL_CHARACTERS
    )
    assert (
        runtime.MAX_SERIALIZED_ROW_CHARACTERS
        == reality.MAX_SERIALIZED_ROW_CHARACTERS
    )


def _opaque(prefix: str) -> str:
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def _view(
    prefix: str,
    *,
    row_count: int = 11,
) -> dict[str, object]:
    return {
        "opaque_item_id": _opaque(prefix),
        "physical_rows": [
            [f"{prefix}-name-{row}", row]
            for row in range(row_count)
        ],
        "question": f"Which Name has Score equal to 7 for {prefix}?",
        "table_header": ["Name", "Score"],
        "table_types": ["text", "real"],
    }


def _label(
    view: dict[str, object],
    *,
    family: str,
    fold_index: int,
    gold_row_ids: tuple[int, ...] = (7,),
    action_view_sha256: str | None = None,
    table_row_count: int | None = None,
) -> dict[str, object]:
    rows = view["physical_rows"]
    assert isinstance(rows, list)
    return {
        "action_view_sha256": (
            runtime.canonical_sha256(view)
            if action_view_sha256 is None
            else action_view_sha256
        ),
        "family": family,
        "fold_index": fold_index,
        "gold_row_ids": list(gold_row_ids),
        "item_commitment_sha256": _opaque(
            "source-" + str(view["opaque_item_id"])
        ),
        "opaque_item_id": view["opaque_item_id"],
        "sqlite_rowid_cross_checked": True,
        "table_row_count": (
            len(rows)
            if table_row_count is None
            else table_row_count
        ),
    }


def _view_pack(block: str, rows: list[dict[str, object]]):
    return runtime.build_view_pack(block=block, items=rows)


def _label_pack(
    views: list[dict[str, object]],
    *,
    families: tuple[str, ...] | None = None,
):
    if families is None:
        families = tuple(
            policy.FAMILY_ORDER[index % len(policy.FAMILY_ORDER)]
            for index in range(len(views))
        )
    labels = [
        _label(
            view,
            family=family,
            fold_index=index % 4,
        )
        for index, (view, family) in enumerate(
            zip(views, families, strict=True)
        )
    ]
    return runtime.build_label_pack(block="A_form", items=labels)


def _rehash(pack: dict[str, object]) -> dict[str, object]:
    base = {
        key: value
        for key, value in pack.items()
        if key != "self_sha256"
    }
    return {**base, "self_sha256": runtime.canonical_sha256(base)}


def test_pack_schemas_match_source_compiler_and_are_content_addressed() -> None:
    views = [_view("alpha"), _view("beta")]
    view_pack = _view_pack("A_form", views)
    assert set(view_pack) == {
        "block",
        "contains_labels",
        "item_count",
        "items",
        "schema",
        "self_sha256",
        "study_id",
    }
    assert view_pack["contains_labels"] is False
    assert view_pack["item_count"] == 2
    assert all(
        set(item)
        == {
            "opaque_item_id",
            "physical_rows",
            "question",
            "table_header",
            "table_types",
        }
        for item in view_pack["items"]
    )
    assert view_pack["self_sha256"] == runtime.canonical_sha256(
        {
            key: value
            for key, value in view_pack.items()
            if key != "self_sha256"
        }
    )
    decoded = runtime.decode_view_pack(
        view_pack, expected_block="A_form", expected_count=2
    )
    assert len(decoded) == 2

    label_pack = _label_pack(views, families=("EQ", "GT"))
    assert set(label_pack) == {
        "block",
        "item_count",
        "items",
        "release_policy",
        "schema",
        "self_sha256",
        "study_id",
    }
    assert all(
        set(item) == runtime._A_FORM_LABEL_ITEM_KEYS
        for item in label_pack["items"]
    )
    labels = runtime.decode_label_pack(
        label_pack, expected_block="A_form", expected_count=2
    )
    assert {label.family for label in labels} == {"EQ", "GT"}

    tampered = json.loads(json.dumps(view_pack))
    tampered["items"][0]["question"] = "tampered"
    with pytest.raises(
        runtime.WikiSQLUAOActionRuntimeError,
        match="content hash",
    ):
        runtime.decode_view_pack(tampered)
    forbidden = _view("forbidden")
    forbidden["family"] = "EQ"
    with pytest.raises(
        runtime.WikiSQLUAOActionRuntimeError,
        match="fields",
    ):
        runtime.build_view_pack(block="A_form", items=[forbidden])


def test_action_runtime_has_no_a_hold_label_capability() -> None:
    parameters = set(inspect.signature(runtime.run_agent).parameters)
    assert parameters == {
        "a_form_view_pack",
        "a_form_label_pack",
        "a_hold_view_pack",
        "encoder",
    }
    assert not parameters.intersection(
        {
            "a_hold_labels",
            "a_hold_label_pack",
            "family",
            "gold",
            "gold_row_ids",
            "sql",
        }
    )
    with pytest.raises(
        runtime.WikiSQLUAOActionRuntimeError,
        match="cannot construct or read A_hold labels",
    ):
        runtime.build_label_pack(block="A_hold", items=[])


def test_raw_actions_are_exact_reality_core_bm25() -> None:
    views = [_view("raw-a"), _view("raw-b")]
    view_pack = _view_pack("A_hold", views)
    decoded = runtime.decode_view_pack(view_pack)
    action_pack = runtime.run_raw(view_pack=view_pack)
    actions = runtime.decode_action_pack(
        action_pack,
        expected_block="A_hold",
        expected_arm="RAW",
        expected_action_view_pack_sha256=view_pack["self_sha256"],
    )
    assert len(actions) == len(decoded) == 2
    by_id = {
        row["opaque_item_id"]: tuple(row["top5_row_ids"])
        for row in actions
    }
    for item in decoded:
        assert by_id[item.item_id] == reality.raw_bm25_top5(
            item.question, item.table
        )
    assert all(
        set(row) == {"opaque_item_id", "top5_row_ids"}
        for row in action_pack["items"]
    )
    assert action_pack["action_view_pack_sha256"] == (
        view_pack["self_sha256"]
    )
    assert action_pack["self_sha256"] == runtime.canonical_sha256(
        {
            key: value
            for key, value in action_pack.items()
            if key != "self_sha256"
        }
    )


class _FakeEncoder:
    model_sha256 = "a" * 64

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def encode(self, texts, *, batch_size):
        values = tuple(texts)
        self.calls.append((values, batch_size))
        return [
            [float(index), float(len(text)), 1.0]
            for index, text in enumerate(values)
        ]


def _compiled_policy() -> policy.CompiledPolicy:
    selected = tuple(
        recipe.claim_id for recipe in policy.CLAIM_RECIPES[:2]
    )
    feature_names = tuple(
        f"{recipe.claim_id}:{name}"
        for recipe in policy.CLAIM_RECIPES[:2]
        for name in recipe.feature_names
    )
    model = policy.LogisticModel(
        feature_names=feature_names,
        population_mean=(0.0,) * len(feature_names),
        population_std=(1.0,) * len(feature_names),
        intercept=-40.0,
        coefficients=(0.0,) * len(feature_names),
    )
    return policy.CompiledPolicy(
        selected_claim_ids=selected,
        model=model,
        probe_receipt_sha256=tuple(
            f"{index}" * 64 for index in range(4)
        ),
        claim_selection_receipt_sha256="a" * 64,
        no_op_calibration_receipt_sha256="b" * 64,
        margin_threshold=0,
        train_item_count=192,
    )


class _Probe:
    def __init__(self, index: int) -> None:
        self.index = index

    def safe_receipt(self):
        base = {"probe_index": self.index, "schema": "fake_probe"}
        return {**base, "self_sha256": policy.canonical_sha256(base)}


class _Formation:
    def __init__(self) -> None:
        self.policy = _compiled_policy()
        self.probe_receipts = tuple(_Probe(index) for index in range(4))
        self.claim_selection_receipt = _Probe(4)
        self.no_op_calibration_receipt = _Probe(5)

    def safe_receipt(self):
        base = {
            "policy_sha256": self.policy.policy_sha256,
            "schema": "fake_formation",
        }
        return {**base, "self_sha256": policy.canonical_sha256(base)}


def test_agent_embedding_alignment_policy_roundtrip_and_action_completeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    form_views = [_view(f"form-{index}") for index in range(3)]
    hold_views = [_view(f"hold-{index}") for index in range(2)]
    form_pack = _view_pack("A_form", form_views)
    labels = _label_pack(
        form_views,
        families=("EQ", "GT", "LT"),
    )
    hold_pack = _view_pack("A_hold", hold_views)
    encoder = _FakeEncoder()
    captured: list[tuple[policy.TrainingItem, ...]] = []
    formation = _Formation()

    def fake_fit(items):
        rows = tuple(items)
        captured.append(rows)
        return formation

    monkeypatch.setattr(policy, "fit_uao_policy", fake_fit)
    artifacts = runtime.run_agent(
        a_form_view_pack=form_pack,
        a_form_label_pack=labels,
        a_hold_view_pack=hold_pack,
        encoder=encoder,
    )
    assert len(encoder.calls) == 1
    texts, batch_size = encoder.calls[0]
    assert batch_size == runtime.ENCODER_BATCH_SIZE
    decoded_form = runtime.decode_view_pack(form_pack)
    decoded_hold = runtime.decode_view_pack(hold_pack)
    assert len(texts) == sum(
        1 + len(item.rows)
        for item in decoded_form + decoded_hold
    )
    assert len(captured) == 1
    assert len(captured[0]) == 3
    cursor = 0
    for training_item, view_item in zip(
        captured[0], decoded_form, strict=True
    ):
        embeddings = training_item.item.embeddings
        assert embeddings is not None
        assert embeddings.question[0] == float(cursor)
        assert embeddings.rows[0][0] == float(cursor + 1)
        cursor += 1 + len(view_item.rows)

    reconstructed = policy.compiled_policy_from_private_payload(
        artifacts.compiled_policy_private
    )
    assert reconstructed.policy_sha256 == formation.policy.policy_sha256
    actions = runtime.decode_action_pack(
        artifacts.action_pack,
        expected_block="A_hold",
        expected_arm="Agent",
        expected_action_view_pack_sha256=hold_pack["self_sha256"],
    )
    assert len(actions) == 2
    assert {row["opaque_item_id"] for row in actions} == {
        item.item_id for item in decoded_hold
    }
    assert all(len(row["top5_row_ids"]) == 5 for row in actions)
    assert artifacts.safe_receipt["a_hold_label_access_count"] == 0
    assert artifacts.safe_receipt["network_call_count"] == 0
    assert artifacts.safe_receipt["retry_count"] == 0
    safe_text = json.dumps(artifacts.action_pack, sort_keys=True)
    assert "question" not in safe_text
    assert "family" not in safe_text
    assert all(
        set(row) == {"opaque_item_id", "top5_row_ids"}
        for row in artifacts.action_pack["items"]
    )


def test_common_action_pack_registry_accepts_only_three_frozen_arms() -> None:
    view_pack = _view_pack("A_hold", [_view("arm-registry")])
    item_id = view_pack["items"][0]["opaque_item_id"]
    item = {
        "opaque_item_id": item_id,
        "top5_row_ids": [0, 1, 2, 3, 4],
    }
    for arm in ("Agent", "RAW", "HippoRAG"):
        pack = runtime.build_action_pack(
            block="A_hold",
            arm=arm,
            action_view_pack_sha256=view_pack["self_sha256"],
            items=[item],
        )
        assert runtime.decode_action_pack(
            pack,
            expected_block="A_hold",
            expected_arm=arm,
            expected_action_view_pack_sha256=(
                view_pack["self_sha256"]
            ),
        ) == (item,)
    with pytest.raises(
        runtime.WikiSQLUAOActionRuntimeError,
        match="action arm",
    ):
        runtime.build_action_pack(
            block="A_hold",
            arm="official_HippoRAG",
            action_view_pack_sha256=view_pack["self_sha256"],
            items=[item],
        )


@pytest.mark.parametrize(
    ("tamper", "expected"),
    (
        ("action_view", "binding"),
        ("row_count", "binding"),
        ("gold_out_of_range", "binding"),
    ),
)
def test_a_form_binding_tamper_fails_before_encoder(
    tamper: str,
    expected: str,
) -> None:
    form_view = _view("bound-form")
    hold_view = _view("bound-hold")
    label = _label(
        form_view,
        family="EQ",
        fold_index=0,
        action_view_sha256=(
            "0" * 64 if tamper == "action_view" else None
        ),
        table_row_count=(
            12 if tamper == "row_count" else None
        ),
        gold_row_ids=(
            (79,) if tamper == "gold_out_of_range" else (7,)
        ),
    )
    form_pack = _view_pack("A_form", [form_view])
    label_pack = runtime.build_label_pack(
        block="A_form", items=[label]
    )
    hold_pack = _view_pack("A_hold", [hold_view])
    encoder = _FakeEncoder()
    with pytest.raises(
        runtime.WikiSQLUAOActionRuntimeError,
        match=expected,
    ):
        runtime.run_agent(
            a_form_view_pack=form_pack,
            a_form_label_pack=label_pack,
            a_hold_view_pack=hold_pack,
            encoder=encoder,
        )
    assert encoder.calls == []


def test_formal_count_family_and_fold_contracts() -> None:
    form_views: list[dict[str, object]] = []
    families: list[str] = []
    labels: list[dict[str, object]] = []
    for family in policy.FAMILY_ORDER:
        for within_family in range(64):
            view = _view(f"formal-{family}-{within_family}")
            form_views.append(view)
            families.append(family)
            labels.append(
                _label(
                    view,
                    family=family,
                    fold_index=within_family % 4,
                )
            )
    hold_views = [_view(f"formal-hold-{index}") for index in range(72)]
    form_pack = _view_pack("A_form", form_views)
    label_pack = runtime.build_label_pack(
        block="A_form", items=labels
    )
    hold_pack = _view_pack("A_hold", hold_views)
    runtime.require_formal_agent_counts(
        a_form_view_pack=form_pack,
        a_form_label_pack=label_pack,
        a_hold_view_pack=hold_pack,
    )
    assert Counter(
        row.family for row in runtime.decode_label_pack(label_pack)
    ) == Counter({"EQ": 64, "GT": 64, "LT": 64})

    with pytest.raises(
        runtime.WikiSQLUAOActionRuntimeError,
        match="item count",
    ):
        runtime.require_formal_agent_counts(
            a_form_view_pack=_view_pack(
                "A_form", form_views[:-1]
            ),
            a_form_label_pack=label_pack,
            a_hold_view_pack=hold_pack,
        )

    imbalanced = list(labels)
    first_lt = next(
        index
        for index, row in enumerate(imbalanced)
        if row["family"] == "LT"
    )
    replacement = dict(imbalanced[first_lt])
    replacement["family"] = "EQ"
    imbalanced[first_lt] = replacement
    with pytest.raises(
        runtime.WikiSQLUAOActionRuntimeError,
        match="family quotas",
    ):
        runtime.require_formal_agent_counts(
            a_form_view_pack=form_pack,
            a_form_label_pack=runtime.build_label_pack(
                block="A_form", items=imbalanced
            ),
            a_hold_view_pack=hold_pack,
        )


def test_exclusive_canonical_output_and_no_overwrite(
    tmp_path: Path,
) -> None:
    action = runtime.run_raw(
        view_pack=_view_pack("A_hold", [_view("exclusive")])
    )
    output = tmp_path / "action.json"
    file_sha256 = runtime._write_exclusive(output, action)
    assert stat.S_ISREG(output.stat().st_mode)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == file_sha256
    assert output.read_bytes() == runtime.canonical_json_bytes(
        action, newline=True
    )
    with pytest.raises(runtime.WikiSQLUAOActionRuntimeError):
        runtime._write_exclusive(output, action)
