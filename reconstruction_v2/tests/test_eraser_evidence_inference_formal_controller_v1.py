from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_direct_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_formal_controller_v1 as subject,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_local_runtime_v1 as local_runtime,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_e3_runner_v1 as runner,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_source_qualification_v1 as source_qualification,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_three_arm_scheduler_v1 as scheduler,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_freeze(project: Path) -> Path:
    rows = []
    for role in acquisition.REQUIRED_IMPLEMENTATION_ROLE_REGISTRY:
        relative = subject.EXPECTED_ROLE_PATHS[role]
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"frozen:{role}\n".encode("ascii")
        path.write_bytes(raw)
        rows.append(
            {
                "relative_path": relative,
                "role": role,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    supplemental_path = project / subject.SUPPLEMENTAL_CONTROLLER_TEST_PATH
    supplemental_path.parent.mkdir(parents=True, exist_ok=True)
    supplemental_raw = b"frozen:formal-controller-lifecycle-test\n"
    supplemental_path.write_bytes(supplemental_raw)
    body = {
        "schema": acquisition.IMPLEMENTATION_FREEZE_SCHEMA,
        "version": "v1",
        "status": "frozen_before_source_qualification_or_private_assignment",
        "design_sha256": acquisition.FORMAL_DESIGN_SHA256,
        "required_role_registry": list(
            acquisition.REQUIRED_IMPLEMENTATION_ROLE_REGISTRY
        ),
        "implementation_binding": {"files": rows},
        "supplemental_controller_test_binding": {
            "relative_path": subject.SUPPLEMENTAL_CONTROLLER_TEST_PATH,
            "sha256": hashlib.sha256(supplemental_raw).hexdigest(),
        },
        "synthetic_test_receipt": {
            "collected_case_count": 7,
            "passed_case_count": 7,
            "real_source_or_benchmark_item_read": False,
            "online_or_network_calls": 0,
        },
    }
    payload = {
        **body,
        acquisition.IMPLEMENTATION_FREEZE_SELF_HASH_FIELD: subject.stable_hash(
            body
        ),
    }
    path = project / subject.FULL_IMPLEMENTATION_FREEZE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(payload))
    return path


def test_full_freeze_verifier_binds_exact_role_path_and_file_hash(
    tmp_path: Path,
) -> None:
    freeze_path = _write_freeze(tmp_path)
    verified = subject.verify_full_implementation_freeze(
        project=tmp_path,
        freeze_path=freeze_path,
    )
    assert verified["design_sha256"] == acquisition.FORMAL_DESIGN_SHA256

    controller = tmp_path / subject.EXPECTED_ROLE_PATHS["formal_controller"]
    controller.write_text("tampered\n", encoding="ascii")
    with pytest.raises(subject.EraserEvidenceInferenceFormalControllerError):
        subject.verify_full_implementation_freeze(
            project=tmp_path,
            freeze_path=freeze_path,
        )


def _view(block: str) -> dict[str, object]:
    items = []
    for ordinal in range(runner.BLOCK_COUNTS[block]):
        items.append(
            {
                "block_ordinal": ordinal,
                "item_commitment_sha256": _sha(f"{block}:{ordinal}"),
                "payload": {
                    "query": f"query {ordinal}",
                    "official_ico": {
                        "Intervention": "intervention",
                        "Comparator": "comparator",
                        "Outcome": "outcome",
                    },
                    "sentence_tokens": [
                        [f"sentence{sentence}", "token"]
                        for sentence in range(5)
                    ],
                },
            }
        )
    return {"block": block, "item_count": len(items), "items": items}


def test_view_conversion_is_exact_and_rejects_label_field() -> None:
    view = _view("A_form")
    rows = subject._view_to_runtime_rows(view, block="A_form")
    assert len(rows) == 48
    assert rows[0].sentence_texts[0] == "sentence0 token"

    view["items"][0]["payload"]["family"] = runner.FAMILIES[0]  # type: ignore[index]
    with pytest.raises(subject.EraserEvidenceInferenceFormalControllerError):
        subject._view_to_runtime_rows(view, block="A_form")


def test_a_form_utility_delta_uses_exact_flattened_union_utility() -> None:
    traces = []
    labels = []
    for ordinal in range(48):
        commitment = _sha(f"delta:{ordinal}")
        traces.append(
            SimpleNamespace(
                item_commitment_sha256=commitment,
                r0_top5=(0, 1, 2, 3, 4),
                r7_top5=(1, 2, 3, 4, 5),
            )
        )
        labels.append(
            runner.AnchorLabel(
                item_commitment_sha256=commitment,
                gold_ordinals=(5,),
                family=runner.FAMILIES[ordinal % 3],
            )
        )
    execution = SimpleNamespace(
        block="A_form",
        feature_seal=SimpleNamespace(
            traces=tuple(traces),
            item_commitments=tuple(row.item_commitment_sha256 for row in traces),
        ),
    )
    observed = subject._a_form_utility_deltas(
        execution=execution,
        labels=labels,
    )
    assert set(observed.values()) == {Fraction(2)}


def test_runtime_preflight_wrapper_rejects_source_or_network_activity() -> None:
    base = {
        "schema": "eraser_evidence_inference_local_runtime_v1_preflight",
        "model_inference_calls": 0,
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
    }
    receipt = subject._runtime_preflight_receipt(base)
    assert receipt["runtime_preflight_receipt_sha256"] == subject.stable_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "runtime_preflight_receipt_sha256"
        }
    )
    with pytest.raises(subject.EraserEvidenceInferenceFormalControllerError):
        subject._runtime_preflight_receipt(
            {**base, "benchmark_source_or_private_pack_reads": 1}
        )


@pytest.mark.parametrize("promoted", [False, True])
def test_promotion_decision_is_only_typed_score_projection(promoted: bool) -> None:
    score = SimpleNamespace(
        block="A_hold",
        evaluator_promoted=promoted,
        score_receipt_sha256=_sha("hold-score"),
    )
    policy = SimpleNamespace(policy_receipt_sha256=_sha("policy"))
    decision = subject._promotion_decision(score=score, policy=policy)
    assert decision["evaluator_promoted"] is promoted
    assert decision["M_search_materialization_authorized"] is promoted
    assert decision["new_threshold_seed_candidate_feature_or_family_added"] is False
    assert decision["promotion_decision_sha256"] == subject.stable_hash(
        {
            key: value
            for key, value in decision.items()
            if key != "promotion_decision_sha256"
        }
    )


def _self_hashed(
    body: dict[str, object], field: str
) -> dict[str, object]:
    return {**body, field: subject.stable_hash(body)}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(payload))


def test_full_synthetic_nonpromotion_lifecycle_wires_all_frozen_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production controller topology without any source/model IO."""

    project = tmp_path / "project"
    controller_root = project / subject.FORMAL_ROOT_RELATIVE / "controller"
    acquisition_root = project / subject.FORMAL_ROOT_RELATIVE / "acquisition"
    controller_root.mkdir(parents=True)
    acquisition_root.mkdir()

    views = {
        block: {
            **_view(block),
            "label_free_view_sha256": _sha(f"view:{block}"),
        }
        for block in runner.BLOCK_ORDER
    }
    public = {
        "source_epoch_marker_sha256": _sha("source-marker"),
        "private_assignment_sha256": _sha("private-assignment"),
        "private_assignment_file_sha256": _sha("private-assignment-file"),
        "public_receipt_sha256": _sha("public-receipt"),
    }
    runtime = object()
    schedule_calls: list[dict[str, object]] = []
    capability_calls: list[dict[str, object]] = []
    label_materializations: list[str] = []
    label_state_loads: list[str] = []
    late_materializations: list[str] = []
    view_loads: list[str] = []
    score_calls: list[dict[str, object]] = []
    m_file_probes: list[str] = []
    packs: dict[str, dict[str, object]] = {}
    schedules_by_wave: list[SimpleNamespace] = []

    def fake_qualification(_project: Path) -> dict[str, object]:
        return _self_hashed(
            {
                "schema": source_qualification.SCHEMA,
                "version": source_qualification.VERSION,
                "status": "passed_source_qualification_no_selection",
                "source_or_item_content_persisted": False,
            },
            "qualification_sha256",
        )

    def fake_acquire_once(**kwargs: object) -> dict[str, object]:
        assert kwargs["selection_secret"] is None
        assert kwargs["enforce_formal_design_identity"] is True
        _write_json(
            acquisition_root / "acquisition.marker.private.json",
            {"source_epoch_marker_sha256": _sha("source-marker")},
        )
        _write_json(
            acquisition_root / "assignment.private.json",
            {"private_assignment_sha256": public["private_assignment_sha256"]},
        )
        _write_json(
            acquisition_root / "acquisition.receipt.json",
            public,
        )
        for block in ("A_form", "F_search"):
            _write_json(
                acquisition_root / "views" / f"{block}.private.json",
                views[block],
            )
        return public

    def fake_load_view(
        *, acquisition_root: Path, block: str, authorization_path: Path | None
    ) -> dict[str, object]:
        del acquisition_root
        view_loads.append(block)
        if block in {"A_form", "F_search"}:
            assert authorization_path is None
        else:
            assert authorization_path is not None
        return views[block]

    def fake_open_runtime(_config: object) -> object:
        return runtime

    def fake_schedule(
        *, items_by_block: dict[str, object], runtime_bundle: object
    ) -> SimpleNamespace:
        assert runtime_bundle is runtime
        counts = {block: len(rows) for block, rows in items_by_block.items()}
        schedule_calls.append(
            {
                "blocks": tuple(items_by_block),
                "counts": counts,
                "logical_task_count": 3 * sum(counts.values()),
                "runtime": runtime_bundle,
            }
        )
        value = SimpleNamespace(
            block_names=tuple(items_by_block),
            rows_by_block={
                block: tuple(rows) for block, rows in items_by_block.items()
            },
        )
        schedules_by_wave.append(value)
        return value

    def fake_persist_schedule(
        *, controller_root: Path, schedule: SimpleNamespace
    ) -> SimpleNamespace:
        wave_name = "_".join(schedule.block_names)
        schedule_receipt = subject.PersistedArtifact(
            path=controller_root / f"{wave_name}.schedule.receipt.json",
            self_sha256=_sha(f"schedule:{wave_name}"),
            file_sha256=_sha(f"schedule-file:{wave_name}"),
        )
        blocks: dict[str, subject.PersistedBlockExecution] = {}
        for block in schedule.block_names:
            traces = tuple(
                SimpleNamespace(
                    item_commitment_sha256=row.item_commitment_sha256,
                    r0_top5=(0, 1, 2, 3, 4),
                    r7_top5=(4, 3, 2, 1, 0),
                )
                for row in schedule.rows_by_block[block]
            )
            feature = SimpleNamespace(
                feature_receipt_sha256=_sha(f"feature:{block}"),
                traces=traces,
                item_commitments=tuple(
                    trace.item_commitment_sha256 for trace in traces
                ),
            )
            artifact = SimpleNamespace(
                block=block,
                feature_seal=feature,
                hippo_retrieval_seal=(
                    SimpleNamespace(name=f"hippo:{block}")
                    if block in {"A_hold", "M_search"}
                    else None
                ),
                raw_retrieval_seal=(
                    SimpleNamespace(name=f"raw:{block}")
                    if block in {"A_hold", "M_search"}
                    else None
                ),
                archive_payload={"archive_sha256": _sha(f"archive:{block}")},
                receipt={"receipt_sha256": _sha(f"execution:{block}")},
                hippo_arm_seal=SimpleNamespace(
                    hipporag_arm_receipt_sha256=_sha(f"hippo-arm:{block}")
                ),
                raw_arm_seal=SimpleNamespace(
                    raw_arm_receipt_sha256=_sha(f"raw-arm:{block}")
                ),
            )
            blocks[block] = subject.PersistedBlockExecution(
                block=block,
                artifact=artifact,
                archive_file_sha256=_sha(f"archive-file:{block}"),
                receipt_file_sha256=_sha(f"execution-file:{block}"),
            )
        return SimpleNamespace(
            artifact=schedule,
            schedule_receipt=schedule_receipt,
            blocks=blocks,
        )

    original_build_capability = acquisition.build_label_capability

    def spy_build_capability(**kwargs: object) -> dict[str, object]:
        capability_calls.append(dict(kwargs))
        return original_build_capability(**kwargs)  # type: ignore[arg-type]

    def fake_materialize_labels(**kwargs: object) -> dict[str, object]:
        block = str(kwargs["block"])
        assert block != "F_search"
        label_materializations.append(block)
        capability_path = Path(kwargs["label_capability_path"])
        capability = json.loads(capability_path.read_text(encoding="ascii"))
        rows = [
            {
                "item_commitment_sha256": row[
                    "item_commitment_sha256"
                ],
                "family": runner.FAMILIES[index % len(runner.FAMILIES)],
                "flattened_gold_sentence_ordinals": [0],
                "validated_groups": [[0]],
            }
            for index, row in enumerate(views[block]["items"])
        ]
        pack = {
            "block": block,
            "item_count": len(rows),
            "items": rows,
            "label_pack_sha256": _sha(f"label-pack:{block}"),
        }
        packs[block] = pack
        _write_json(
            acquisition_root / "labels" / f"{block}.private.json", pack
        )
        _write_json(
            acquisition_root
            / "stage_markers"
            / f"label.{block}.private.json",
            {"label_stage_marker_sha256": _sha(f"label-marker:{block}")},
        )
        _write_json(
            acquisition_root / "authorizations" / f"label.{block}.private.json",
            capability,
        )
        return pack

    def fake_label_state(
        *,
        acquisition_root: Path,
        block: str,
        label_capability_path: Path | None = None,
    ) -> dict[str, object]:
        del acquisition_root
        label_state_loads.append(block)
        assert label_capability_path is not None
        capability = json.loads(
            label_capability_path.read_text(encoding="ascii")
        )
        capability_raw = subject.canonical_bytes(capability)
        return {
            "block": block,
            "label_pack_sha256": packs[block]["label_pack_sha256"],
            "label_capability_sha256": capability[
                "label_capability_sha256"
            ],
            "label_capability_file_sha256": hashlib.sha256(
                capability_raw
            ).hexdigest(),
            "label_stage_marker_sha256": _sha(f"label-marker:{block}"),
            "upstream_typed_artifact_content_verified_by_acquisition": False,
        }

    def fake_fit_e3(**kwargs: object) -> SimpleNamespace:
        assert kwargs["fold_secret"] == b"f" * 32
        assert kwargs["feature_seal"].feature_receipt_sha256 == _sha(
            "feature:A_form"
        )
        assert len(kwargs["utility_deltas"]) == runner.BLOCK_COUNTS["A_form"]
        receipt = _self_hashed(
            {"schema": f"{runner.VERSION}_e3_fit_receipt"},
            "fit_receipt_sha256",
        )
        return SimpleNamespace(
            receipt=receipt,
            fit_receipt_sha256=receipt["fit_receipt_sha256"],
        )

    def fake_freeze_policy(**kwargs: object) -> SimpleNamespace:
        assert kwargs["feature_seal"].feature_receipt_sha256 == _sha(
            "feature:F_search"
        )
        receipt = _self_hashed(
            {"schema": f"{runner.VERSION}_policy_receipt"},
            "policy_receipt_sha256",
        )
        return SimpleNamespace(
            receipt=receipt,
            policy_receipt_sha256=receipt["policy_receipt_sha256"],
        )

    def fake_materialize_late(**kwargs: object) -> dict[str, object]:
        block = str(kwargs["block"])
        assert block == "A_hold"
        late_materializations.append(block)
        _write_json(
            acquisition_root / "views" / f"{block}.private.json",
            views[block],
        )
        _write_json(
            acquisition_root
            / "stage_markers"
            / f"view.{block}.private.json",
            _self_hashed(
                {"schema": f"{acquisition.VERSION}_stage_marker"},
                "stage_marker_sha256",
            ),
        )
        return views[block]

    def fake_score_anchor(**kwargs: object) -> SimpleNamespace:
        score_calls.append(dict(kwargs))
        assert kwargs["block"] == "A_hold"
        assert kwargs["hippo_retrieval_seal"].name == "hippo:A_hold"
        assert kwargs["raw_retrieval_seal"].name == "raw:A_hold"
        receipt = _self_hashed(
            {
                "schema": f"{runner.VERSION}_A_hold_score_receipt",
                "A_hold_real_domain_primary_passed": False,
                "RAW_block_passed": False,
            },
            "score_receipt_sha256",
        )
        return SimpleNamespace(
            block="A_hold",
            evaluator_promoted=False,
            receipt=receipt,
            score_receipt_sha256=receipt["score_receipt_sha256"],
        )

    def fake_verify_base(**kwargs: object) -> dict[str, object]:
        assert kwargs["enforce_formal_design_identity"] is True
        return {"public_receipt_sha256": public["public_receipt_sha256"]}

    def fail_full_verifier(**_kwargs: object) -> dict[str, object]:
        pytest.fail("nonpromotion must not invoke the verifier that stats M_search")

    original_sha256_file = subject._sha256_file

    def spy_sha256_file(path: Path, field: str) -> str:
        if "M_search" in str(path):
            m_file_probes.append(str(path))
        return original_sha256_file(path, field)

    monkeypatch.setattr(
        source_qualification, "build_formal_qualification", fake_qualification
    )
    monkeypatch.setattr(acquisition, "acquire_once", fake_acquire_once)
    monkeypatch.setattr(acquisition, "load_verified_block_view", fake_load_view)
    monkeypatch.setattr(
        acquisition, "build_label_capability", spy_build_capability
    )
    monkeypatch.setattr(
        acquisition, "materialize_label_pack_once", fake_materialize_labels
    )
    monkeypatch.setattr(
        acquisition, "load_verified_label_state", fake_label_state
    )
    monkeypatch.setattr(
        acquisition, "derive_a_form_fold_key", lambda **_kwargs: b"f" * 32
    )
    monkeypatch.setattr(
        acquisition, "materialize_late_view_once", fake_materialize_late
    )
    monkeypatch.setattr(
        acquisition, "verify_acquisition_state", fake_verify_base
    )
    monkeypatch.setattr(
        acquisition, "verify_full_acquisition_state", fail_full_verifier
    )
    monkeypatch.setattr(local_runtime, "open_runtime", fake_open_runtime)
    monkeypatch.setattr(scheduler, "run_three_arm_schedule", fake_schedule)
    monkeypatch.setattr(subject, "_persist_schedule", fake_persist_schedule)
    monkeypatch.setattr(runner, "fit_e3", fake_fit_e3)
    monkeypatch.setattr(runner, "freeze_f_policy", fake_freeze_policy)
    monkeypatch.setattr(runner, "score_anchor", fake_score_anchor)
    monkeypatch.setattr(subject, "_sha256_file", spy_sha256_file)

    lifecycle = subject.PersistedArtifact(
        path=controller_root / subject.MARKER_FILENAME,
        self_sha256=_sha("lifecycle-marker"),
        file_sha256=_sha("lifecycle-marker-file"),
    )
    preflight = subject.PersistedArtifact(
        path=controller_root / "runtime.preflight.receipt.json",
        self_sha256=_sha("preflight"),
        file_sha256=_sha("preflight-file"),
    )
    stage_state = {"name": "synthetic_start"}
    result = subject._run_started_lifecycle(
        project=project,
        controller_root=controller_root,
        acquisition_root=acquisition_root,
        runtime_config=SimpleNamespace(),
        lifecycle_marker=lifecycle,
        preflight_artifact=preflight,
        stage_state=stage_state,
    )

    assert schedule_calls[0] == {
        "blocks": ("A_form", "F_search"),
        "counts": {"A_form": 48, "F_search": 36},
        "logical_task_count": 252,
        "runtime": runtime,
    }
    assert [call["blocks"] for call in schedule_calls] == [
        ("A_form", "F_search"),
        ("A_hold",),
    ]
    assert all(call["runtime"] is runtime for call in schedule_calls)
    assert [call["block"] for call in capability_calls] == [
        "A_form",
        "A_hold",
    ]
    for call in capability_calls:
        block = str(call["block"])
        assert call["three_arm_execution_seal_sha256"] == _sha(
            f"execution:{block}"
        )
        assert call["feature_seal_sha256"] == _sha(f"feature:{block}")
    assert label_materializations == ["A_form", "A_hold"]
    assert "F_search" not in label_state_loads
    assert late_materializations == ["A_hold"]
    assert not any(block == "M_search" for block in view_loads)
    assert len(score_calls) == 1
    assert m_file_probes == []
    assert not any("M_search" in path.name for path in project.rglob("*"))
    assert result["status"] == "complete_nonpromotion_M_search_unopened"
    assert result["M_search_label_free_view_sha256"] is None
    assert result["M_search_score_receipt"] is None
    private_ledger = result["artifact_file_bindings"][
        "private_materialized_files"
    ]
    assert "source_epoch_marker" in private_ledger
    assert "A_hold_view_stage_marker" in private_ledger
    assert "M_search_view_stage_marker" not in private_ledger
    assert result["terminal_result_sha256"] == subject.stable_hash(
        {
            key: value
            for key, value in result.items()
            if key != "terminal_result_sha256"
        }
    )
    assert (controller_root / subject.RESULT_FILENAME).is_file()
    assert stage_state["name"] == "postflight_and_terminal_result"
