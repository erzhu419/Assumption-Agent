from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import tempfile

import pytest

from replication_runtime.maud_extraction_p1_official_v1 import (
    worker as official_worker,
)
from assumption_agent.benchmarks import (
    maud_extraction_p1_coordinate_worker_v1 as coordinate_worker,
)
from assumption_agent.benchmarks import (
    maud_extraction_p1_formal_controller_v1 as subject,
)
from assumption_agent.benchmarks import maud_extraction_p1_runtime_v1 as runtime
from assumption_agent.benchmarks import (
    maud_extraction_p1_typed_core_v1 as core,
)


@pytest.fixture
def tmp_path() -> Path:
    """Use the Linux filesystem because the frozen contract audits POSIX mode."""

    path = Path(tempfile.mkdtemp(prefix="maud-p1-controller-", dir="/tmp"))
    os.chmod(path, 0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _runtime_paths(tmp_path: Path) -> runtime.RuntimePaths:
    values = {
        name: str((tmp_path / name).absolute())
        for name in runtime.RuntimePaths.__dataclass_fields__
    }
    return runtime.RuntimePaths(**values)


def _frozen_canary_receipt(
    runtime_fingerprint_self_sha256: str,
) -> dict[str, object]:
    return subject.self_hashed(
        {
            "schema": subject.FULL_CANARY_SCHEMA,
            "study_id": subject.STUDY_ID,
            "study_design_self_sha256": (
                subject.STUDY_DESIGN_SELF_SHA256
            ),
            "pre_source_clarification_self_sha256": (
                subject.PRE_SOURCE_CLARIFICATION_SELF_SHA256
            ),
            "status": (
                "passed_source_free_coordinate_pair_and_"
                "official_hipporag"
            ),
            "runtime_fingerprint_sha256": (
                runtime_fingerprint_self_sha256
            ),
            "coordinate_fixture_sha256": "1" * 64,
            "minilm_output_self_sha256": "2" * 64,
            "cross_encoder_output_self_sha256": "3" * 64,
            "official_output_sha256": "4" * 64,
            "official_safe_terminal_sha256": "5" * 64,
            "typed_recipe_registry_sha256": "6" * 64,
            "E0_behavior_set_sha256": "7" * 64,
            "shape": {
                "coordinate_contract_count": 1,
                "coordinate_query_count": 22,
                "coordinate_worker_count": 2,
                "official_contract_count": 1,
                "typed_recipe_count_per_query": len(core.RECIPE_IDS),
                "E0_selection_count": 22,
            },
            "execution": {
                "coordinate_workers_bulk_submitted_before_join": True,
                "cross_encoder_physical_gpu": "1",
                "minilm_physical_gpu": "0",
                "official_physical_gpu": "0",
                "retry_replay_resample_count": 0,
                "api_or_online_evaluator_call_count": 0,
                "formal_source_action_or_score_count": 0,
            },
        }
    )


def _context() -> str:
    return subject._reconstruct_context(  # noqa: SLF001
        subject.synthetic_canary_fixture().passages
    )


def _action_view(block: str, *, contracts: int = 1) -> dict[str, object]:
    split = {
        subject.BLOCK_A_FORM: "TRAIN",
        subject.BLOCK_F_SEARCH: "TRAIN",
        subject.BLOCK_A_HOLD: "DEV",
        subject.BLOCK_M_SEARCH: "TEST",
    }[block]
    context = _context()
    rows = []
    for contract_index in range(contracts):
        contract_id = hashlib.sha256(
            f"{block}-contract-{contract_index}".encode("ascii")
        ).hexdigest()
        items = []
        for item_index in range(22):
            items.append(
                {
                    "work_id": hashlib.sha256(
                        (
                            f"{block}-contract-{contract_index}-"
                            f"item-{item_index}"
                        ).encode("ascii")
                    ).hexdigest(),
                    "question": (
                        "Which clause concerns the public synthetic "
                        f"topic {item_index}?"
                    ),
                    "deal_point_type": f"Synthetic type {item_index:02d}",
                    "family": core.QUERY_FAMILIES[
                        item_index % len(core.QUERY_FAMILIES)
                    ],
                }
            )
        rows.append(
            {
                "contract_work_id": contract_id,
                "context": context,
                "context_sha256": hashlib.sha256(
                    context.encode("utf-8")
                ).hexdigest(),
                "items": items,
            }
        )
    return {
        "schema": subject.ACTION_VIEW_SCHEMA,
        "study_id": subject.STUDY_ID,
        "split": split,
        "block": block,
        "contract_count": len(rows),
        "item_count": len(rows) * 22,
        "contracts": rows,
        "answerability_gold_text_offset_or_span_included": False,
    }


def _write_action_view(
    tmp_path: Path, block: str, *, contracts: int = 1
) -> tuple[Path, subject.ArchiveBinding]:
    path = tmp_path / f"{block}.action.private.json"
    binding = subject.write_private_json_once(
        path, _action_view(block, contracts=contracts)
    )
    return path, subject.ArchiveBinding.from_mapping(binding)


class FakeCoordinates:
    def __init__(self) -> None:
        self.inputs: list[MappingLike] = []

    def __call__(
        self,
        *,
        private_input,
        stage_root,
        runtime_paths,
    ) -> subject.CoordinateBatchResult:
        del stage_root, runtime_paths
        checked = coordinate_worker.validate_private_input(private_input)
        self.inputs.append(private_input)
        minilm_rows = []
        cross_rows = []
        pairwise = []
        for contract in checked:
            passage_count = len(contract["passages"])
            scores = [
                max(0, 900_000 - ordinal * 80_000)
                for ordinal in range(passage_count)
            ]
            for query in contract["queries"]:
                minilm_rows.append(
                    {"work_id": query["work_id"], "scores": scores}
                )
                cross_rows.append(
                    {"work_id": query["work_id"], "scores": scores}
                )
            pairwise.append(
                {
                    "contract_id": contract["contract_id"],
                    "pairwise_scores": [
                        [
                            1_000_000 if left == right else 500_000
                            for right in range(passage_count)
                        ]
                        for left in range(passage_count)
                    ],
                }
            )
        input_sha = coordinate_worker.semantic_sha256(private_input)
        minilm = coordinate_worker.coordinate_output(
            role=coordinate_worker.ROLE_MINILM,
            rows=minilm_rows,
            input_sha256=input_sha,
            model_tree_sha256=str(
                runtime.EXPECTED_MINILM_TREE["tree_sha256"]
            ),
            contract_pairwise=pairwise,
        )
        cross = coordinate_worker.coordinate_output(
            role=coordinate_worker.ROLE_CROSS_ENCODER,
            rows=cross_rows,
            input_sha256=input_sha,
            model_tree_sha256=str(
                runtime.EXPECTED_CROSS_ENCODER_TREE["tree_sha256"]
            ),
        )
        return subject.CoordinateBatchResult(
            minilm_output=minilm, cross_encoder_output=cross
        )


MappingLike = dict[str, object]


class FakeHippo:
    def __init__(self, coordinates: FakeCoordinates) -> None:
        self.coordinates = coordinates
        self.jobs = ()

    def __call__(self, jobs, *, runtime_paths):
        del runtime_paths
        self.jobs = tuple(jobs)
        outputs = []
        coordinate_contracts = self.coordinates.inputs[-1]["contracts"]
        for job, coordinate_contract in zip(
            self.jobs, coordinate_contracts
        ):
            contract_id, corpus_hash, documents, queries = (
                official_worker.validate_input(job.payload)
            )
            assert [row.text for row in documents] == [
                row["text"]
                for row in coordinate_contract["passages"]
            ]
            assert [(row.work_id, row.text) for row in queries] == [
                (row["work_id"], row["question"])
                for row in coordinate_contract["queries"]
            ]
            output = official_worker._output_payload(  # noqa: SLF001
                contract_work_id=contract_id,
                corpus_hash=corpus_hash,
                passage_count=len(documents),
                queries=queries,
                top5_rows=[
                    tuple(range(core.TOP_K)) for _ in queries
                ],
                graph_nodes=len(documents),
                graph_edges=1,
            )
            outputs.append(
                runtime.WorkerRun(
                    output=output,
                    safe_terminal={"safe_phase": "synthetic_passed"},
                )
            )
        return tuple(outputs)


def _empty_gold(
    action_view: dict[str, object],
    *,
    block: str,
    answerable: bool = False,
) -> dict[str, object]:
    contracts = []
    for contract in action_view["contracts"]:
        contracts.append(
            {
                "contract_work_id": contract["contract_work_id"],
                "items": [
                    {
                        "work_id": item["work_id"],
                        "spans": (
                            [{"start": 0, "end": 7, "text": "Section"}]
                            if answerable
                            else []
                        ),
                        "merged_intervals": (
                            [[0, 7]] if answerable else []
                        ),
                    }
                    for item in contract["items"]
                ],
            }
        )
    body = {
        "schema": subject.GOLD_PACK_SCHEMA,
        "study_id": subject.STUDY_ID,
        "split": {
            subject.BLOCK_A_FORM: "TRAIN",
            subject.BLOCK_A_HOLD: "DEV",
            subject.BLOCK_M_SEARCH: "TEST",
        }[block],
        "block": block,
        "contract_count": len(contracts),
        "item_count": len(contracts) * 22,
        "contracts": contracts,
    }
    return {**body, "gold_pack_sha256": subject.semantic_sha256(body)}


def _write_gold(
    tmp_path: Path,
    action_view: dict[str, object],
    *,
    block: str,
    answerable: bool = False,
) -> tuple[Path, subject.ArchiveBinding]:
    value = _empty_gold(
        action_view, block=block, answerable=answerable
    )
    path = tmp_path / f"{block}.gold.sealed.private.json"
    binding = subject.write_private_json_once(path, value)
    return path, subject.ArchiveBinding(
        file_sha256=str(binding["file_sha256"]),
        semantic_sha256=str(value["gold_pack_sha256"]),
        size_bytes=int(binding["size_bytes"]),
    )


def _promotion_receipt(*, promoted: bool) -> dict[str, object]:
    net = Fraction(1, 4) if promoted else Fraction(0)
    body = {
        "schema": subject.PROMOTION_SCHEMA,
        "study_id": subject.STUDY_ID,
        "study_design_self_sha256": subject.STUDY_DESIGN_SELF_SHA256,
        "source_custody_self_sha256": (
            subject.SOURCE_CUSTODY_SELF_SHA256
        ),
        "initial_acquisition_receipt_self_sha256": "1" * 64,
        "A_hold_action_archive_file_sha256": "2" * 64,
        "A_hold_action_archive_semantic_sha256": "3" * 64,
        "A_hold_gold_file_sha256": "4" * 64,
        "A_hold_gold_semantic_sha256": "5" * 64,
        "incumbent_evaluator_id": "E0_FIXED_GENERAL_COVERAGE",
        "challenger_evaluator_id": "E1_AFORM_CENTERED_RIDGE_L2_1",
        "challenger_model_sha256": "6" * 64,
        "challenger_model_self_sha256": "7" * 64,
        "E1_minus_E0_comparison": {
            "contract_count": 4,
            "nonzero_contract_count": 4 if promoted else 0,
            "net": subject._fraction_payload(net),  # noqa: SLF001
            "exact_sign_flip_reference_tail": (
                subject._fraction_payload(Fraction(1, 16))  # noqa: SLF001
            ),
        },
        "promoted": promoted,
        "M_search_authorized": promoted,
        "promotion_rule": (
            "net_strictly_positive_and_complete_contract_sign_flip_"
            "reference_tail_at_most_1_over_10"
        ),
        "retry_replay_resample_refit_or_gate_change_count": 0,
        "online_evaluator_API_or_fine_tune_count": 0,
    }
    return subject.self_hashed(body)


def test_controller_has_no_source_or_acquisition_import_and_no_gold_action_api() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        "maud_extraction_p1_source_v1" in name
        or "maud_extraction_p1_acquisition_v1" in name
        for name in imported
    )
    assert "gold" not in inspect.signature(
        subject.run_action_stage
    ).parameters
    with pytest.raises(
        subject.MaudExtractionP1FormalControllerError,
        match="structurally forbidden",
    ):
        subject.load_acquisition_archive(
            Path("/never/read"),
            block=subject.BLOCK_F_SEARCH,
            role="gold",
        )


def test_full_canary_uses_one_built_passage_corpus_for_all_workers(
    tmp_path: Path,
) -> None:
    fixture = subject.synthetic_canary_fixture()
    context = subject._reconstruct_context(fixture.passages)  # noqa: SLF001
    assert core.build_passages(context) == fixture.passages
    coordinate_contract = fixture.coordinate_input["contracts"][0]
    assert [
        row["text"] for row in coordinate_contract["passages"]
    ] == [
        row["text"] for row in fixture.official_payload["documents"]
    ]
    assert len(fixture.passages) >= 5

    coordinates = FakeCoordinates()

    def official_launcher(**kwargs):
        payload = kwargs["payload"]
        contract_id, corpus_hash, documents, queries = (
            official_worker.validate_input(payload)
        )
        return runtime.WorkerRun(
            output=official_worker._output_payload(  # noqa: SLF001
                contract_work_id=contract_id,
                corpus_hash=corpus_hash,
                passage_count=len(documents),
                queries=queries,
                top5_rows=[
                    tuple(range(core.TOP_K)) for _ in queries
                ],
                graph_nodes=len(documents),
                graph_edges=1,
            ),
            safe_terminal={"safe_phase": "passed"},
        )

    receipt = subject.run_full_source_free_canary(
        runtime_paths=_runtime_paths(tmp_path),
        runtime_fingerprint_sha256="f" * 64,
        canary_root=tmp_path / "canary",
        coordinate_launcher=coordinates,
        official_launcher=official_launcher,
    )
    assert receipt["shape"]["coordinate_worker_count"] == 2
    assert receipt["shape"]["typed_recipe_count_per_query"] == 9
    assert receipt["shape"]["E0_selection_count"] == 22


def test_coordinate_environment_is_typed_only_clean_and_thread_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-escape")
    monkeypatch.setenv("RUOLI_SECRET_TOKEN", "must-not-escape")
    private_root = tmp_path / "coordinate_env"
    private_root.mkdir(mode=0o700)
    paths = _runtime_paths(tmp_path)
    environment = subject._coordinate_environment(  # noqa: SLF001
        paths, physical_gpu="1", private_root=private_root
    )

    assert "OPENAI_API_KEY" not in environment
    assert "RUOLI_SECRET_TOKEN" not in environment
    assert environment["PYTHONPATH"] == paths.typed_pythonpath()
    assert paths.overlay_root not in environment["PYTHONPATH"]
    assert paths.hipporag_source_root not in environment["PYTHONPATH"]
    assert paths.p16_site_root not in environment["PYTHONPATH"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert all(
        environment[key] == value
        for key, value in runtime.CPU_THREAD_ENV.items()
    )


def test_a_hold_actions_use_identical_serialization_and_two_serial_gpu_lanes(
    tmp_path: Path,
) -> None:
    view_path, view_binding = _write_action_view(
        tmp_path, subject.BLOCK_A_HOLD, contracts=3
    )
    coordinates = FakeCoordinates()
    hippo = FakeHippo(coordinates)

    # A zero-target A_form fit is a valid frozen model and deterministically
    # exercises the E1 selection interface.
    base_view_path, base_binding = _write_action_view(
        tmp_path, subject.BLOCK_A_FORM
    )
    base_stage = subject.run_action_stage(
        block=subject.BLOCK_A_FORM,
        action_view_path=base_view_path,
        action_view_binding=base_binding,
        acquisition_receipt_self_sha256="a" * 64,
        stage_root=tmp_path / "base_action",
        runtime_paths=_runtime_paths(tmp_path),
        coordinate_launcher=coordinates,
        hippo_batch_launcher=hippo,
    )
    base_view = _action_view(subject.BLOCK_A_FORM)
    gold_path, gold_binding = _write_gold(
        tmp_path, base_view, block=subject.BLOCK_A_FORM
    )
    model, _ = subject.score_a_form(
        action_archive_path=base_stage.archive_path,
        gold_path=gold_path,
        gold_binding=gold_binding,
        output_root=tmp_path / "base_score",
    )
    stage = subject.run_action_stage(
        block=subject.BLOCK_A_HOLD,
        action_view_path=view_path,
        action_view_binding=view_binding,
        acquisition_receipt_self_sha256="b" * 64,
        stage_root=tmp_path / "hold_action",
        runtime_paths=_runtime_paths(tmp_path),
        e1_model_path=model.model_path,
        coordinate_launcher=coordinates,
        hippo_batch_launcher=hippo,
    )
    archive = subject.read_canonical_private(stage.archive_path)
    assert archive["gold_open_count_before_archive"] == 0
    assert [job.physical_gpu for job in hippo.jobs] == ["0", "1", "0"]
    assert len(archive["contracts"]) == 3

    hold_view = _action_view(subject.BLOCK_A_HOLD, contracts=3)
    hold_gold_path, hold_gold_binding = _write_gold(
        tmp_path,
        hold_view,
        block=subject.BLOCK_A_HOLD,
        answerable=True,
    )
    promotion, hold_score = subject.score_a_hold(
        action_archive_path=stage.archive_path,
        gold_path=hold_gold_path,
        gold_binding=hold_gold_binding,
        e1_model_path=model.model_path,
        initial_acquisition_receipt_self_sha256="b" * 64,
        source_custody_self_sha256=(
            subject.SOURCE_CUSTODY_SELF_SHA256
        ),
        output_root=tmp_path / "hold_score",
    )
    comparison = promotion["E1_minus_E0_comparison"]
    derived = (
        subject._fraction_from_payload(comparison["net"]) > 0  # noqa: SLF001
        and subject._fraction_from_payload(  # noqa: SLF001
            comparison["exact_sign_flip_reference_tail"]
        )
        <= Fraction(1, 10)
    )
    assert promotion["promoted"] is derived
    secondary = hold_score.receipt["safe_aggregates"][
        "secondary_metrics_non_gate"
    ]
    assert set(secondary) == {
        subject.ROLE_AGENT_E0,
        subject.ROLE_AGENT_E1,
        subject.ROLE_RAW,
        subject.ROLE_HIPPORAG,
    }
    assert all(
        row["complete_at_5_count"] == 3 * 22
        and row["coverage_at_least_half_count"] == 3 * 22
        for row in secondary.values()
    )


def test_a_form_unanswerable_rows_are_retained_as_zero_targets(
    tmp_path: Path,
) -> None:
    view = _action_view(subject.BLOCK_A_FORM)
    view_path = tmp_path / "A_form.action.private.json"
    raw_binding = subject.write_private_json_once(view_path, view)
    stage = subject.run_action_stage(
        block=subject.BLOCK_A_FORM,
        action_view_path=view_path,
        action_view_binding=subject.ArchiveBinding.from_mapping(
            raw_binding
        ),
        acquisition_receipt_self_sha256="a" * 64,
        stage_root=tmp_path / "actions",
        runtime_paths=_runtime_paths(tmp_path),
        coordinate_launcher=FakeCoordinates(),
        hippo_batch_launcher=lambda jobs, **kwargs: (),
    )
    gold_path, gold_binding = _write_gold(
        tmp_path, view, block=subject.BLOCK_A_FORM
    )
    model, score = subject.score_a_form(
        action_archive_path=stage.archive_path,
        gold_path=gold_path,
        gold_binding=gold_binding,
        output_root=tmp_path / "score",
    )
    loaded, _, _ = subject.load_e1_model(model.model_path)
    assert loaded.training_row_count == 22 * len(core.RECIPE_IDS)
    aggregates = score.receipt["safe_aggregates"]
    assert aggregates["training_item_count"] == 22
    assert aggregates["answerable_training_item_count"] == 0
    assert aggregates["unanswerable_item_count"] == 22


def test_promotion_is_derived_and_nonpromotion_never_launches_test(
    tmp_path: Path,
) -> None:
    nonpromotion = _promotion_receipt(promoted=False)
    subject.validate_promotion_receipt(
        nonpromotion,
        expected_initial_receipt_self_sha256="1" * 64,
        expected_action_archive_file_sha256="2" * 64,
        expected_action_archive_semantic_sha256="3" * 64,
        expected_gold_file_sha256="4" * 64,
        expected_gold_semantic_sha256="5" * 64,
        expected_model_sha256="6" * 64,
        expected_model_self_sha256="7" * 64,
        require_promoted=False,
    )
    tampered = dict(nonpromotion)
    tampered["promoted"] = True
    tampered["M_search_authorized"] = True
    tampered = subject.self_hashed(
        {key: value for key, value in tampered.items() if key != "self_sha256"}
    )
    with pytest.raises(
        subject.MaudExtractionP1FormalControllerError,
        match="not derived",
    ):
        subject.validate_promotion_receipt(
            tampered,
            expected_initial_receipt_self_sha256="1" * 64,
            expected_action_archive_file_sha256="2" * 64,
            expected_action_archive_semantic_sha256="3" * 64,
            expected_gold_file_sha256="4" * 64,
            expected_gold_semantic_sha256="5" * 64,
            expected_model_sha256="6" * 64,
            expected_model_self_sha256="7" * 64,
            require_promoted=False,
        )

    acquisition_root = tmp_path / "acquisition"
    acquisition_root.mkdir(mode=0o700)
    promotion_path = tmp_path / "nonpromotion.json"
    subject.write_private_json_once(promotion_path, nonpromotion)
    calls = []

    def never_runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("TEST subprocess must remain closed")

    with pytest.raises(
        subject.MaudExtractionP1FormalControllerError,
        match="not derived",
    ):
        subject.run_test_acquisition_process(
            paths=_runtime_paths(tmp_path),
            test_path=tmp_path / "test.json",
            secret_path=tmp_path / "secret",
            download_receipt_path=tmp_path / "download.receipt.json",
            acquisition_root=acquisition_root,
            promotion_receipt_path=promotion_path,
            process_root=tmp_path / "test_process",
            expected_initial_receipt_self_sha256="1" * 64,
            expected_action_archive_file_sha256="2" * 64,
            expected_action_archive_semantic_sha256="3" * 64,
            expected_gold_file_sha256="4" * 64,
            expected_gold_semantic_sha256="5" * 64,
            expected_model_sha256="6" * 64,
            expected_model_self_sha256="7" * 64,
            runner=never_runner,
        )
    assert calls == []


def test_promoted_receipt_unlocks_exactly_one_test_subprocess(
    tmp_path: Path,
) -> None:
    promotion = _promotion_receipt(promoted=True)
    acquisition_root = tmp_path / "acquisition"
    acquisition_root.mkdir(mode=0o700)
    promotion_path = tmp_path / "promotion.json"
    subject.write_private_json_once(promotion_path, promotion)
    calls = []

    class Completed:
        returncode = 0

    def runner(command, **kwargs):
        calls.append(tuple(command))
        receipt = subject.self_hashed(
            {
                "schema": (
                    "maud_extraction_p1_acquisition_v1_"
                    "test_receipt_v1"
                ),
                "study_id": subject.STUDY_ID,
                "status": (
                    "promotion_authorized_TEST_parse_complete_"
                    "and_process_must_exit"
                ),
                "a_hold_promotion_receipt_sha256": promotion[
                    "self_sha256"
                ],
                "private_archives": {},
                "retry_replay_resample_or_secret_rotation_count": 0,
            }
        )
        subject.write_private_json_once(
            acquisition_root / "test_parse.receipt.json", receipt
        )
        return Completed()

    subject.run_test_acquisition_process(
        paths=_runtime_paths(tmp_path),
        test_path=tmp_path / "test.json",
        secret_path=tmp_path / "secret",
        download_receipt_path=tmp_path / "download.receipt.json",
        acquisition_root=acquisition_root,
        promotion_receipt_path=promotion_path,
        process_root=tmp_path / "test_process",
        expected_initial_receipt_self_sha256="1" * 64,
        expected_action_archive_file_sha256="2" * 64,
        expected_action_archive_semantic_sha256="3" * 64,
        expected_gold_file_sha256="4" * 64,
        expected_gold_semantic_sha256="5" * 64,
        expected_model_sha256="6" * 64,
        expected_model_self_sha256="7" * 64,
        runner=runner,
    )
    assert len(calls) == 1
    assert calls[0][1:5] == (
        "-S",
        "-B",
        "-m",
        subject.ACQUISITION_MODULE,
    )
    assert "test" in calls[0]
    assert "--download-receipt" in calls[0]
    assert calls[0][calls[0].index("--download-receipt") + 1] == str(
        tmp_path / "download.receipt.json"
    )


def test_formal_reads_exact_frozen_fingerprint_and_canary_without_rerun(
    tmp_path: Path,
) -> None:
    paths = _runtime_paths(tmp_path)
    fingerprint = subject.self_hashed(
        {
            "schema": runtime.RUNTIME_FINGERPRINT_SCHEMA,
            "study_id": subject.STUDY_ID,
            "study_design_self_sha256": (
                subject.STUDY_DESIGN_SELF_SHA256
            ),
            "pre_source_clarification_self_sha256": (
                subject.PRE_SOURCE_CLARIFICATION_SELF_SHA256
            ),
            "status": (
                "verified_source_free_post_reboot_runtime_fingerprint"
            ),
            "path_commitments": paths.path_commitments(),
        }
    )
    fingerprint_path = tmp_path / "runtime_fingerprint.json"
    subject.write_private_json_once(fingerprint_path, fingerprint)
    fingerprint_sha256 = str(fingerprint["self_sha256"])
    assert (
        subject._validate_runtime_fingerprint(  # noqa: SLF001
            fingerprint_path,
            paths,
            expected_self_sha256=fingerprint_sha256,
        )
        == fingerprint_sha256
    )
    with pytest.raises(
        subject.MaudExtractionP1FormalControllerError,
        match="fingerprint drifted",
    ):
        subject._validate_runtime_fingerprint(  # noqa: SLF001
            fingerprint_path,
            paths,
            expected_self_sha256="f" * 64,
        )

    canary = _frozen_canary_receipt(fingerprint_sha256)
    canary_path = tmp_path / "full_canary.receipt.json"
    subject.write_private_json_once(canary_path, canary)
    assert (
        subject._validate_full_source_free_canary_receipt(  # noqa: SLF001
            canary_path,
            expected_self_sha256=str(canary["self_sha256"]),
            expected_runtime_fingerprint_self_sha256=(
                fingerprint_sha256
            ),
        )
        == canary
    )
    with pytest.raises(
        subject.MaudExtractionP1FormalControllerError,
        match="execution-freeze binding",
    ):
        subject._validate_full_source_free_canary_receipt(  # noqa: SLF001
            canary_path,
            expected_self_sha256="e" * 64,
            expected_runtime_fingerprint_self_sha256=(
                fingerprint_sha256
            ),
        )
    formal_source = inspect.getsource(subject.run_formal_study)
    assert "run_full_source_free_canary" not in formal_source


def test_formal_failure_writes_one_content_free_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_root = tmp_path / "formal"
    control_root.mkdir(mode=0o700)
    config = subject.FormalExecutionConfig(
        train_path=tmp_path / "train.json",
        dev_path=tmp_path / "dev.json",
        test_path=tmp_path / "test.json",
        secret_path=tmp_path / "secret",
        download_receipt_path=tmp_path / "download.receipt.json",
        acquisition_root=tmp_path / "acquisition",
        control_root=control_root,
        runtime_fingerprint_path=tmp_path / "fingerprint.json",
        full_canary_receipt_path=tmp_path / "canary.json",
        expected_runtime_fingerprint_self_sha256="a" * 64,
        expected_full_canary_self_sha256="b" * 64,
        runtime_paths=_runtime_paths(tmp_path),
    )

    def fail_once(*args, **kwargs):
        raise RuntimeError("PRIVATE CONTRACT TEXT MUST NOT LEAK")

    monkeypatch.setattr(subject, "_run_formal_study_once", fail_once)
    with pytest.raises(RuntimeError, match="PRIVATE CONTRACT"):
        subject.run_formal_study(config)
    failure_path = control_root / "formal.failure.terminal.json"
    first_bytes = failure_path.read_bytes()
    receipt = subject.read_canonical_private(failure_path)
    subject.verify_self_hash(receipt)
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["safe_phase"] == "formal_control_initialized"
    assert (
        receipt[
            "raw_contract_item_action_gold_score_or_error_message_included"
        ]
        is False
    )
    assert b"PRIVATE CONTRACT" not in first_bytes

    with pytest.raises(RuntimeError, match="PRIVATE CONTRACT"):
        subject.run_formal_study(config)
    assert failure_path.read_bytes() == first_bytes
