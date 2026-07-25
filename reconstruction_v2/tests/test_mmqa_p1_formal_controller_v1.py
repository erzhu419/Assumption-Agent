from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import pytest

from assumption_agent.benchmarks import mmqa_p1_action_integration_v1 as integration
from assumption_agent.benchmarks import mmqa_p1_formal_controller_v1 as controller
from assumption_agent.benchmarks import mmqa_p1_private_selection_v1 as selection
from assumption_agent.benchmarks import mmqa_p1_typed_proof_e5_core_v1 as core


@dataclass(frozen=True)
class SyntheticSelection:
    root: Path
    receipt: dict[str, Any]


def _work_id(block: str, index: int) -> str:
    digest = hashlib.sha256(f"{block}:{index}".encode("ascii")).hexdigest()
    return f"mmqa-work-v1-{digest}"


def _action_item(block: str, index: int) -> dict[str, Any]:
    nodes = [
        {
            "ordinal": ordinal,
            "node_type": core.ROW if ordinal < 3 else core.TEXT,
            "content": f"anonymous {block} item {index} unit {ordinal}",
        }
        for ordinal in range(6)
    ]
    edges = []
    for row, text in ((0, 3), (1, 4), (2, 5)):
        edges.extend(
            (
                {
                    "source_ordinal": row,
                    "target_ordinal": text,
                    "edge_type": core.ROW_TO_TEXT,
                },
                {
                    "source_ordinal": text,
                    "target_ordinal": row,
                    "edge_type": core.TEXT_TO_ROW,
                },
            )
        )
    return {
        "work_id": _work_id(block, index),
        "question": f"Resolve anonymous {block} item {index}",
        "nodes": nodes,
        "edges": edges,
    }


def _gold_item(
    block: str,
    index: int,
    *,
    a_hold_pair: tuple[int, int],
    m_pair: tuple[int, int],
) -> dict[str, Any]:
    row, text = (
        a_hold_pair
        if block == "A_hold"
        else m_pair
        if block == "M_search"
        else (0, 3)
    )
    value: dict[str, Any] = {
        "work_id": _work_id(block, index),
        "gold_row_ordinals": [row],
        "gold_text_ordinals": [text],
        "exact_gold_pairs": [
            {"row_ordinal": row, "text_ordinal": text}
        ],
    }
    if block == "A_form":
        value["oof_fold"] = index % 5
    elif block in {"A_hold", "M_search"}:
        quota = selection.BLOCK_QUOTA_PER_FAMILY[block]
        value["evaluation_family"] = selection.FAMILIES[index // quota]
    return value


def _write_pack(
    path: Path, value: dict[str, Any], *, semantic_field: str
) -> dict[str, Any]:
    binding = selection._atomic_write_json(path, value, mode=0o600)  # noqa: SLF001
    return {
        **binding,
        "relative_path": path.name,
        "semantic_sha256": value[semantic_field],
    }


def _synthetic_selection(
    tmp_path: Path,
    *,
    a_hold_pair: tuple[int, int] = (2, 5),
    m_pair: tuple[int, int] = (2, 5),
) -> SyntheticSelection:
    root = tmp_path / "selection"
    root.mkdir(mode=0o700)
    pack_bindings: dict[str, Any] = {}
    a_form_commitment_rows: list[dict[str, Any]] = []
    for block in selection.BLOCK_ORDER:
        count = selection.BLOCK_ITEM_COUNTS[block]
        action_items = [_action_item(block, index) for index in range(count)]
        action_body = {
            "schema": f"{selection.VERSION}_label_free_action_pack_v1",
            "version": selection.VERSION,
            "study_id": selection.STUDY_ID,
            "block": block,
            "item_count": count,
            "item_exact_fields": ["work_id", "question", "nodes", "edges"],
            "source_identifier_family_exact_type_answer_support_or_metadata_included": False,
            "items": action_items,
        }
        action = selection.self_hashed(action_body, "action_pack_sha256")
        gold_items = [
            _gold_item(
                block,
                index,
                a_hold_pair=a_hold_pair,
                m_pair=m_pair,
            )
            for index in range(count)
        ]
        if block == "A_form":
            a_form_commitment_rows = [
                {"work_id": row["work_id"], "oof_fold": row["oof_fold"]}
                for row in gold_items
            ]
        gold_body = {
            "schema": f"{selection.VERSION}_sealed_gold_pack_v1",
            "version": selection.VERSION,
            "study_id": selection.STUDY_ID,
            "block": block,
            "item_count": count,
            "action_pack_sha256": action["action_pack_sha256"],
            "source_identifier_exact_type_answer_or_support_included": False,
            "evaluation_family_included_as_late_only_scoring_stratum": (
                block in {"A_hold", "M_search"}
            ),
            "evaluation_family_forbidden_from_action_E5_features_fit_or_policy": True,
            "component_atomic_HMAC_oof_fold_included": block == "A_form",
            "items": gold_items,
        }
        gold = selection.self_hashed(gold_body, "gold_pack_sha256")
        action_path = root / selection.ACTION_PACK_FILENAMES[block]
        gold_path = root / selection.GOLD_PACK_FILENAMES[block]
        pack_bindings[block] = {
            "action": _write_pack(
                action_path, action, semantic_field="action_pack_sha256"
            ),
            "gold": _write_pack(
                gold_path, gold, semantic_field="gold_pack_sha256"
            ),
        }
    fold_sizes = {
        str(fold): sum(row["oof_fold"] == fold for row in a_form_commitment_rows)
        for fold in range(5)
    }
    receipt_body = {
        "schema": f"{selection.VERSION}_public_receipt_v1",
        "version": selection.VERSION,
        "study_id": selection.STUDY_ID,
        "status": "private_one_shot_selection_complete",
        "binding_self_sha256": {
            "source_custody": selection.SOURCE_CUSTODY_SELF_SHA256,
            "study_design": selection.STUDY_DESIGN_SELF_SHA256,
        },
        "selection_contract": {
            "A_form_five_fold_OOF": {
                "fold_count": 5,
                "component_atomic": True,
                "secret_HMAC_ordered_deterministic_balancing": True,
                "fold_sizes": fold_sizes,
                "assignment_commitment_sha256": controller.stable_hash(
                    a_form_commitment_rows
                ),
            }
        },
        "private_pack_bindings": pack_bindings,
        "model_network_retrieval_evaluator_or_score_calls": 0,
        "retry_replay_resample_or_secret_rotation": 0,
        "source_item_identifier_content_answer_or_support_published": False,
    }
    receipt = selection.self_hashed(receipt_body, "acquisition_sha256")
    selection._assert_public_safe(receipt)  # noqa: SLF001
    selection._atomic_write_json(  # noqa: SLF001
        root / selection.PUBLIC_RECEIPT_FILENAME, receipt, mode=0o644
    )
    return SyntheticSelection(root, receipt)


class Coordinates:
    def __init__(self) -> None:
        self.blocks: list[str] = []

    def __call__(
        self,
        *,
        block: str,
        items: dict[str, integration.AnonymousWorkItem],
    ) -> dict[str, tuple[integration.UnitCoordinates, ...]]:
        self.blocks.append(block)
        scores = {
            0: 0.95,
            1: 0.60,
            2: 0.10,
            3: 0.90,
            4: 0.50,
            5: 0.05,
        }
        return {
            work_id: tuple(
                integration.UnitCoordinates(
                    ordinal=unit.ordinal,
                    minilm_similarity=scores[unit.ordinal],
                    cross_encoder_relevance=scores[unit.ordinal],
                    entity_anchor=0,
                    relation_anchor=0,
                    numeric_or_temporal_anchor=0,
                )
                for unit in item.units
            )
            for work_id, item in items.items()
        }


class Hippo:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, *, block: str, payloads: dict[str, Any]) -> dict[str, list[int]]:
        self.calls += 1
        assert block == "A_hold"
        if self.fail:
            raise RuntimeError("synthetic Hippo failure")
        return {work_id: [0, 3, 1, 4, 2] for work_id in payloads}


def _fixed_model(slates: Any, *, low_bundle: bool) -> core.E5Model:
    slates = tuple(slates)
    coefficients = [-1.0, -1.0, -1.0, -1.0] + [0.0] * 7
    if not low_bundle:
        coefficients = [0.0] * len(core.FEATURE_ORDER)
    return core.E5Model(
        population_mean=(0.0,) * len(core.FEATURE_ORDER),
        population_std=(1.0,) * len(core.FEATURE_ORDER),
        coefficients=tuple(coefficients),
        training_item_count=len(slates),
        training_bundle_count=sum(
            len(slate.work.actions.bundles) for slate in slates
        ),
        solver="numpy_deterministic_lbfgs_m10_v1",
        iterations=0,
        converged=True,
        objective=0.0,
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_valid_promotion_reality_l5_f_sealed_and_archive_before_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic = _synthetic_selection(tmp_path)
    control_root = tmp_path / "controller"
    coordinates = Coordinates()
    hippo = Hippo()
    monkeypatch.setattr(
        controller,
        "_fit_prepared_slates",
        lambda slates: _fixed_model(slates, low_bundle=True),
    )
    opened: list[str] = []
    original_open = selection.open_block_gold

    def checked_open(**kwargs: Any) -> dict[str, Any]:
        block = str(kwargs["block"])
        archive = (
            control_root
            / "stages"
            / block
            / controller.STAGE_ACTION_ARCHIVE_FILENAME
        )
        assert archive.is_file()
        assert _mode(archive) == 0o600
        json.loads(archive.read_text("ascii"))
        opened.append(block)
        return original_open(**kwargs)

    monkeypatch.setattr(selection, "open_block_gold", checked_open)
    final = controller.run_lifecycle(
        selection_root=synthetic.root,
        control_root=control_root,
        expected_selection_acquisition_sha256=synthetic.receipt[
            "acquisition_sha256"
        ],
        coordinate_provider=coordinates,
        hippo_executor=hippo,
    )

    assert final["A_hold"]["promoted"] is True
    assert final["A_hold"]["reality_primary_passed"] is True
    assert final["M_search"]["L5_passed"] is True
    assert final["M_search"]["gold_opened"] is True
    assert final["F_search"]["gold_opened"] is False
    assert final["F_search"]["used_as_gate"] is False
    assert opened == ["A_form", "A_hold", "M_search"]
    assert coordinates.blocks == ["A_form", "F_search", "A_hold", "M_search"]
    assert hippo.calls == 1
    target_audit = json.loads(
        (
            control_root
            / "stages"
            / "A_form"
            / controller.STAGE_SCORE_FILENAME
        ).read_text("ascii")
    )
    assert target_audit["target_rule"] == (
        "all_presealed_bundles_containing_at_least_one_late_exact_gold_"
        "row_text_pair_are_positive_multiple_positives_use_logsumexp_"
        "bundle_first_nDCG_is_audit_only"
    )
    assert target_audit["exact_positive_slate_count"] == 120
    assert (
        target_audit[
            "no_exact_positive_omitted_conditional_slate_count"
        ]
        == 0
    )
    for item in target_audit["items"]:
        for candidate in item["sealed_bundle_first_candidates"]:
            assert candidate["positive_exact_gold_bundle_target"] is (
                candidate["bundle_contains_exact_gold_pair"]
            )
            assert "admissible_maximum_utility_target" not in candidate
    model_receipt = json.loads(
        (control_root / controller.FULL_MODEL_FILENAME).read_text("ascii")
    )
    assert model_receipt["population_scaler_slate_count"] == 120
    assert model_receipt["training_item_count"] == 120
    assert model_receipt["exact_positive_slate_count"] == 120
    assert (
        model_receipt[
            "no_exact_positive_omitted_conditional_slate_count"
        ]
        == 0
    )
    assert not (
        synthetic.root / selection.GOLD_OPEN_MARKER_FILENAMES["F_search"]
    ).exists()
    for block in ("A_form", "A_hold", "M_search"):
        authorization = json.loads(
            (
                control_root
                / "stages"
                / block
                / controller.GOLD_AUTHORIZATION_FILENAME
            ).read_text("ascii")
        )
        expected_archive = (
            control_root
            / "stages"
            / block
            / controller.STAGE_ACTION_ARCHIVE_FILENAME
        ).absolute()
        assert authorization["action_archive_paths"] == [
            str(expected_archive)
        ]
        assert len(authorization["action_archive_semantic_sha256s"]) == 1
    m_authorization = json.loads(
        (
            control_root
            / "stages"
            / "M_search"
            / controller.GOLD_AUTHORIZATION_FILENAME
        ).read_text("ascii")
    )
    a_hold_score_path = (
        control_root
        / "stages"
        / "A_hold"
        / controller.STAGE_SCORE_FILENAME
    )
    a_hold_archive_path = (
        control_root
        / "stages"
        / "A_hold"
        / controller.STAGE_ACTION_ARCHIVE_FILENAME
    )
    assert m_authorization["A_hold_promotion_receipt_path"] == str(
        a_hold_score_path.absolute()
    )
    assert m_authorization["A_hold_promotion_action_archive_path"] == str(
        a_hold_archive_path.absolute()
    )

    forged_body = json.loads(a_hold_score_path.read_text("ascii"))
    forged_body.pop("score_sha256")
    forged_body["status"] = "valid_nonpromotion_M_search_sealed"
    forged_body["promoted"] = False
    forged_body["M_search_authorized"] = False
    forged_promotion = selection.self_hashed(forged_body, "score_sha256")
    forged_promotion_path = tmp_path / "forged.A_hold.score.private.json"
    selection._atomic_write_json(  # noqa: SLF001
        forged_promotion_path, forged_promotion, mode=0o600
    )
    with pytest.raises(
        selection.MmqaP1PrivateSelectionError,
        match="promotion receipt drifted",
    ):
        selection.write_block_gold_open_authorization(
            tmp_path / "forged.M.authorization.private.json",
            output_root=synthetic.root,
            block="M_search",
            action_archive_sha256s=(
                final["M_search"]["action_archive_file_sha256"],
            ),
            action_archive_paths=(
                control_root
                / "stages"
                / "M_search"
                / controller.STAGE_ACTION_ARCHIVE_FILENAME,
            ),
            promotion_sha256=forged_promotion["score_sha256"],
            promotion_receipt_path=forged_promotion_path,
            promotion_action_archive_path=a_hold_archive_path,
        )
    selection._assert_public_safe(final)  # noqa: SLF001
    for path in control_root.rglob("*"):
        assert _mode(path) == (0o700 if path.is_dir() else 0o600)


def test_valid_nonpromotion_never_materializes_or_opens_m(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic = _synthetic_selection(tmp_path, a_hold_pair=(0, 3))
    control_root = tmp_path / "controller"
    coordinates = Coordinates()
    hippo = Hippo()
    monkeypatch.setattr(
        controller,
        "_fit_prepared_slates",
        lambda slates: _fixed_model(slates, low_bundle=False),
    )
    final = controller.run_lifecycle(
        selection_root=synthetic.root,
        control_root=control_root,
        expected_selection_acquisition_sha256=synthetic.receipt[
            "acquisition_sha256"
        ],
        coordinate_provider=coordinates,
        hippo_executor=hippo,
    )

    assert final["A_hold"]["promoted"] is False
    assert final["M_search"] == {
        "status": "sealed_after_A_hold_valid_nonpromotion",
        "authorized": False,
        "gold_opened": False,
        "action_archive_created": False,
        "L5_passed": False,
    }
    assert coordinates.blocks == ["A_form", "F_search", "A_hold"]
    assert not (control_root / "stages" / "M_search").exists()
    assert not (
        synthetic.root / selection.GOLD_OPEN_MARKER_FILENAMES["M_search"]
    ).exists()
    assert not (
        synthetic.root / selection.GOLD_OPEN_MARKER_FILENAMES["F_search"]
    ).exists()


def test_hippo_failure_is_terminal_without_gold_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic = _synthetic_selection(tmp_path)
    control_root = tmp_path / "controller"
    coordinates = Coordinates()
    hippo = Hippo(fail=True)
    monkeypatch.setattr(
        controller,
        "_fit_prepared_slates",
        lambda slates: _fixed_model(slates, low_bundle=True),
    )
    with pytest.raises(
        controller.MmqaP1FormalControllerError,
        match="HippoRAG batch failed",
    ):
        controller.run_lifecycle(
            selection_root=synthetic.root,
            control_root=control_root,
            expected_selection_acquisition_sha256=synthetic.receipt[
                "acquisition_sha256"
            ],
            coordinate_provider=coordinates,
            hippo_executor=hippo,
        )

    failure = json.loads(
        (control_root / controller.FAILURE_FILENAME).read_text("ascii")
    )
    assert failure["status"] == "terminal_consumed_no_retry_replay_or_resample"
    assert failure["failed_phase"] == "A_hold_seal_all_four_arm_actions"
    assert _mode(control_root / controller.FAILURE_FILENAME) == 0o600
    assert not (control_root / controller.FINAL_RECEIPT_FILENAME).exists()
    assert hippo.calls == 1
    assert not (
        synthetic.root / selection.GOLD_OPEN_MARKER_FILENAMES["A_hold"]
    ).exists()
    assert not (
        synthetic.root / selection.GOLD_OPEN_MARKER_FILENAMES["M_search"]
    ).exists()
    with pytest.raises(
        controller.MmqaP1FormalControllerError, match="root already exists"
    ):
        controller.run_lifecycle(
            selection_root=synthetic.root,
            control_root=control_root,
            expected_selection_acquisition_sha256=synthetic.receipt[
                "acquisition_sha256"
            ],
            coordinate_provider=coordinates,
            hippo_executor=hippo,
        )
    assert hippo.calls == 1


def test_full_ndcg_without_exact_pair_is_never_a_positive_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = controller._convert_action_item(  # noqa: SLF001
        _action_item("A_form", 900)
    )
    work_id = loaded.work_id
    work_item = loaded.work_item
    coordinates = Coordinates()(
        block="A_form", items={work_id: work_item}
    )[work_id]
    actions = integration.form_actions(work_item, coordinates)
    work = controller.WorkExecution(
        work_id=work_id,
        work_item=work_item,
        coordinates=coordinates,
        actions=actions,
        bundle_first_top5=controller._bundle_top5_rows(  # noqa: SLF001
            actions
        ),
    )
    pairs = (
        integration.ExactRowTextLink(0, 3),
        integration.ExactRowTextLink(1, 4),
    )
    positives = controller._positive_exact_gold_bundles(  # noqa: SLF001
        actions.bundles, pairs
    )
    assert core.ProofBundle((0, 3)) in positives
    assert core.ProofBundle((1, 4)) in positives
    assert len(positives) >= 2

    no_pair = core.ProofBundle((2, 5))
    synthetic_full_ndcg_top5 = (0, 1, 3, 4, 2)
    assert core.binary_evidence_ndcg_at_5(
        synthetic_full_ndcg_top5, (0, 1, 3, 4)
    ) == 1.0
    assert not controller._bundle_contains_exact_gold_pair(  # noqa: SLF001
        no_pair, pairs
    )
    assert no_pair not in positives
    slate = controller.TrainingSlate(
        work=work,
        fold=0,
        gold_evidence=(0, 1, 3, 4),
        exact_gold_pairs=pairs,
        positive_exact_gold_bundles=positives,
        maximum_bundle_first_integer_utility_audit_only=(
            core.integer_binary_evidence_utility(
                synthetic_full_ndcg_top5, (0, 1, 3, 4)
            )
        ),
        neutral_no_exact_gold_bundle=False,
    )
    observed_gold_sizes: list[tuple[int, ...]] = []
    original_objective = core._conditional_loss_gradient  # noqa: SLF001

    def recording_objective(
        beta: Any,
        feature_slates: Any,
        gold_indices: Any,
    ) -> Any:
        observed_gold_sizes.append(
            tuple(len(indices) for indices in gold_indices)
        )
        return original_objective(beta, feature_slates, gold_indices)

    monkeypatch.setattr(
        core, "_conditional_loss_gradient", recording_objective
    )
    controller._fit_prepared_slates((slate,))  # noqa: SLF001
    assert observed_gold_sizes
    assert all(sizes == (len(positives),) for sizes in observed_gold_sizes)
    assert len(positives) >= 2

    with pytest.raises(
        controller.MmqaP1FormalControllerError,
        match="exact-gold training target drifted",
    ):
        controller.TrainingSlate(
            work=work,
            fold=0,
            gold_evidence=(0, 1, 3, 4),
            exact_gold_pairs=pairs,
            positive_exact_gold_bundles=(no_pair,),
            maximum_bundle_first_integer_utility_audit_only=(
                core.integer_binary_evidence_utility(
                    synthetic_full_ndcg_top5, (0, 1, 3, 4)
                )
            ),
            neutral_no_exact_gold_bundle=False,
        )


def test_gold_outside_retained_closure_scores_normally() -> None:
    rows = tuple(
        integration.SerializedUnit(index, f"row {index}", core.ROW)
        for index in range(49)
    )
    texts = (
        integration.SerializedUnit(49, "retained text", core.TEXT),
        integration.SerializedUnit(50, "outside-pair text", core.TEXT),
    )
    links = (
        integration.ExactRowTextLink(0, 49),
        integration.ExactRowTextLink(1, 49),
        integration.ExactRowTextLink(48, 50),
    )
    item = integration.AnonymousWorkItem(
        "Gold outside the row-capped closure", rows, texts, links
    )
    coordinates = tuple(
        integration.UnitCoordinates(
            unit.ordinal,
            0.0 if unit.ordinal == 48 else 1.0,
            0.0 if unit.ordinal == 48 else 1.0,
            0,
            0,
            0,
        )
        for unit in item.units
    )
    actions = integration.form_actions(item, coordinates)
    assert 48 not in actions.shared_closure.ordinals
    work = controller.WorkExecution(
        work_id=_work_id("outside", 0),
        work_item=item,
        coordinates=coordinates,
        actions=actions,
        bundle_first_top5=controller._bundle_top5_rows(actions),  # noqa: SLF001
    )
    archive_item = controller._stage_archive_item(work)  # noqa: SLF001
    selection._validate_action_archive_item(  # noqa: SLF001
        archive_item,
        expected_work_id=work.work_id,
        expected_projection_sha256=item.anonymous_projection_sha256,
        expected_row_ordinals=tuple(range(49)),
        expected_text_ordinals=(49, 50),
        block="A_form",
    )
    tampered_archive_item = json.loads(json.dumps(archive_item))
    tampered_archive_item["coordinates"][48][
        "minilm_similarity_float64_hex"
    ] = (2.0).hex()
    tampered_archive_item["coordinates"][48][
        "cross_encoder_relevance_float64_hex"
    ] = (2.0).hex()
    tampered_archive_item["coordinate_vector_sha256"] = selection.stable_hash(
        tampered_archive_item["coordinates"]
    )
    with pytest.raises(
        selection.MmqaP1PrivateSelectionError,
        match="action feature archive drifted",
    ):
        selection._validate_action_archive_item(  # noqa: SLF001
            tampered_archive_item,
            expected_work_id=work.work_id,
            expected_projection_sha256=item.anonymous_projection_sha256,
            expected_row_ordinals=tuple(range(49)),
            expected_text_ordinals=(49, 50),
            block="A_form",
        )
    gold, pairs = controller._gold_projection(  # noqa: SLF001
        work,
        {
            "gold_row_ordinals": [48],
            "gold_text_ordinals": [50],
            "exact_gold_pairs": [
                {"row_ordinal": 48, "text_ordinal": 50}
            ],
        },
    )
    scores = integration.score_late_gold(
        actions, gold, exact_gold_pairs=pairs
    )
    assert scores.e0.integer_utility >= 0
    assert scores.raw.integer_utility >= 0
    utilities = tuple(
        core.integer_binary_evidence_utility(top5, gold)
        for _bundle, top5 in work.bundle_first_top5
    )
    assert len(work.bundle_first_top5) >= 2
    assert set(utilities) == {0}
    neutral = controller.TrainingSlate(
        work=work,
        fold=0,
        gold_evidence=gold,
        exact_gold_pairs=pairs,
        positive_exact_gold_bundles=(),
        maximum_bundle_first_integer_utility_audit_only=0,
        neutral_no_exact_gold_bundle=True,
    )
    model = controller._fit_prepared_slates((neutral,))  # noqa: SLF001
    assert model.training_item_count == 1
    assert model.training_bundle_count == len(actions.bundles)
    assert model.coefficients == (0.0,) * len(core.FEATURE_ORDER)
