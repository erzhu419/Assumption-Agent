from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from typing import Iterator

import pytest

from assumption_agent.benchmarks import feverous_e2_evaluator_v1 as evaluator
from assumption_agent.benchmarks import hybridqa_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks import hybridqa_p6_e2_formal_controller_v1 as controller
from assumption_agent.benchmarks import hybridqa_query_anchored_formal_runner_v1 as runner


FREEZE_SHA256 = "f" * 64
SECRET = b"S" * 32


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _self_hashed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: controller.stable_hash(body)}


def _write_private_json(path: Path, value: dict[str, object]) -> str:
    raw = _canonical_bytes(value, newline=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _candidate(index: int, family: str) -> acquisition.Candidate:
    table = f"{family}-table-{index}"
    target = f"/wiki/p-{index}"
    if family == "TABLE_ONLY":
        gold = (acquisition.UnitKey("table_row", table, "0"),)
    elif family == "PASSAGE_ONLY":
        gold = (
            acquisition.UnitKey("linked_passage", table, target),
            acquisition.UnitKey("table_row", table, "0"),
        )
    else:
        gold = (
            acquisition.UnitKey("linked_passage", table, target),
            acquisition.UnitKey("table_row", table, "0"),
            acquisition.UnitKey("table_row", table, "1"),
        )
    return acquisition.Candidate(
        source_ordinal=index,
        question_id=f"{family}-q-{index}",
        table_id=table,
        question=f"Which synthetic value {index}",
        question_postag="WDT JJ NN CD",
        family=family,
        gold_unit_keys=tuple(sorted(gold)),
    )


def _candidate_pool() -> tuple[acquisition.Candidate, ...]:
    return tuple(
        _candidate(index, family)
        for family in acquisition.FAMILIES
        for index in range(60)
    )


def _unit_stream(
    selected: dict[str, tuple[acquisition.Candidate, ...]],
) -> tuple[acquisition.CorpusUnit, ...]:
    units: dict[acquisition.UnitKey, acquisition.CorpusUnit] = {}
    for block in acquisition.BLOCK_ORDER:
        for candidate in selected[block]:
            target = next(
                (
                    key.local_key
                    for key in candidate.gold_unit_keys
                    if key.unit_type == "linked_passage"
                ),
                None,
            )
            for key in candidate.gold_unit_keys:
                if key.unit_type == "linked_passage":
                    unit = acquisition.CorpusUnit(
                        key,
                        f"passage {candidate.table_id}",
                        f"passage body {candidate.source_ordinal}",
                        None,
                        (key.local_key,),
                    )
                else:
                    row = int(key.local_key)
                    links = (target,) if target is not None and row == 0 else ()
                    unit = acquisition.CorpusUnit(
                        key,
                        f"table {candidate.table_id}",
                        f"row {row} body {candidate.source_ordinal}",
                        row,
                        links,
                    )
                units[key] = unit
    distractor = 0
    while len(units) < 900:
        key = acquisition.UnitKey("table_row", f"distractor-{distractor}", "0")
        units[key] = acquisition.CorpusUnit(
            key,
            f"distractor title {distractor}",
            f"distractor body {distractor}",
            0,
            (),
        )
        distractor += 1
    return tuple(units.values())


def _source_qualification_receipt() -> dict[str, object]:
    source = acquisition.source_qualification
    code_sha256 = hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()
    body: dict[str, object] = {
        "schema": source.SCHEMA,
        "version": source.VERSION,
        "source_release": source.SOURCE_RELEASE,
        "qualification_class": source.QUALIFICATION_CLASS,
        "status": "source_qualified_for_embedded_pre_secret_acquisition",
        "formal_identity_enforced": True,
        "qualification_code_sha256": code_sha256,
        "file_sets": {
            "qa_required_file_count": len(source.QA_RELATIVE_PATHS),
            "qa_required_file_set_sha256": "1" * 64,
            "table_request_pair_count": source.FORMAL_CORPUS_COUNT,
            "table_request_file_set_sha256": "2" * 64,
        },
        "qa": {
            "train_row_count": source.FORMAL_QA_COUNTS["train"],
            "dev_row_count": source.FORMAL_QA_COUNTS["dev"],
            "test_row_count": source.FORMAL_QA_COUNTS["test"],
            "question_id_count": sum(source.FORMAL_QA_COUNTS.values()),
            "question_ids_unique_within_splits": True,
            "question_id_splits_pairwise_disjoint": True,
            "train_traced_raw_exact_match": True,
            "dev_traced_raw_exact_match": True,
            "train_empty_answer_node_row_count": 0,
            "dev_empty_answer_node_row_count": 0,
            "referenced_table_count": 100,
        },
        "dev_reference": {
            "question_id_count": source.FORMAL_QA_COUNTS["dev"],
            "reference_answer_exact_match": True,
            "table_partition_count": source.FORMAL_DEV_REFERENCE_PARTITION["table"],
            "passage_partition_count": source.FORMAL_DEV_REFERENCE_PARTITION["passage"],
            "computed_partition_count": source.FORMAL_DEV_REFERENCE_PARTITION["computed"],
            "partition_complete_and_disjoint": True,
        },
        "corpus": {
            "table_json_count": source.FORMAL_CORPUS_COUNT,
            "request_json_count": source.FORMAL_CORPUS_COUNT,
            "table_request_filename_sets_equal": True,
            "dataset_table_ids_exactly_resolved": True,
            "unused_table_count": source.FORMAL_CORPUS_COUNT - 100,
            "data_row_count": 1000,
            "header_and_data_cell_count": 2000,
            "link_reference_count": 300,
            "request_entry_count": 300,
            "empty_request_entry_count": 0,
            "empty_request_link_reference_count": 0,
            "all_links_exactly_resolved": True,
        },
        "answer_nodes": {
            "answer_node_count": 30,
            "table_source_count": 10,
            "passage_source_count": 20,
            "sources_coordinates_and_links_valid": True,
        },
        "source_custody": {
            "clean_checkout_verified_before_and_after": True,
            "hybridqa": {
                "commit": source.FORMAL_HYBRIDQA_COMMIT,
                "tree": source.FORMAL_HYBRIDQA_TREE,
                "tracked_file_count": 1,
                "tracked_file_set_sha256": "3" * 64,
            },
            "wikitables_with_links": {
                "commit": source.FORMAL_WIKITABLES_COMMIT,
                "tree": source.FORMAL_WIKITABLES_TREE,
                "tracked_file_count": 1,
                "tracked_file_set_sha256": "4" * 64,
            },
        },
        "safeguards": {
            "pre_design_programmatic_audit_occurred": True,
            "pre_design_programmatic_audit_raw_output_count": 0,
            "raw_record_output_count": 0,
            "per_row_or_linkable_hash_output_count": 0,
            "selection_secret_created_or_read_count": 0,
            "selection_or_hmac_count": 0,
            "action_or_retrieval_count": 0,
            "score_or_utility_count": 0,
            "dev_test_online_evaluator_count": 0,
            "standalone_qualification_manifest_persisted_count": 0,
        },
    }
    return _self_hashed(body, "receipt_sha256")


@pytest.fixture
def private_tmp_path(tmp_path: Path) -> Iterator[Path]:
    # Codex App can place pytest's configured tmp_path on DrvFS, whose chmod
    # emulation cannot represent the controller's mandatory 0700/0600
    # capabilities.  Keep the fixture fully synthetic and temporary while
    # exercising the production Linux permission contract.
    linux_tmp = Path("/tmp")
    parent = str(linux_tmp) if linux_tmp.is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix=f"hybridqa-controller-{tmp_path.name}-", dir=parent
    ) as value:
        yield Path(value)


@pytest.fixture
def synthetic_project(
    private_tmp_path: Path,
) -> Iterator[tuple[Path, controller.AcquisitionBoundary]]:
    tmp_path = private_tmp_path
    selected = acquisition.select_blocks(_candidate_pool(), secret=SECRET)
    corpus, mapping = acquisition.form_fixed_corpus(
        selected=selected,
        unit_stream=_unit_stream(selected),
        secret=SECRET,
    )
    packs = acquisition.form_private_packs(
        selected=selected,
        corpus=corpus,
        unit_to_index=mapping,
    )

    formal_root = tmp_path / acquisition.FORMAL_ROOT_RELATIVE
    acquisition_root = tmp_path / acquisition.ACQUISITION_RELATIVE
    acquisition_root.mkdir(mode=0o700, parents=True)
    formal_root.chmod(0o700)
    acquisition_root.chmod(0o700)

    file_hashes = {
        filename: _write_private_json(acquisition_root / filename, payload)
        for filename, payload in packs.items()
    }
    secret_path = acquisition_root / acquisition.SECRET_FILENAME
    secret_path.write_bytes(SECRET)
    secret_path.chmod(0o600)
    marker = _self_hashed(
        {
            "schema": f"{acquisition.VERSION}_one_shot_marker",
            "version": acquisition.VERSION,
            "status": "formal_attempt_started",
            "design_sha256": acquisition.DESIGN_SHA256,
            "implementation_freeze_sha256": FREEZE_SHA256,
            "source_validation_completed": False,
            "selection_secret_created": False,
        },
        "marker_sha256",
    )
    _write_private_json(acquisition_root / acquisition.MARKER_FILENAME, marker)
    type_counts = Counter(unit.key.unit_type for unit in corpus)
    public = _self_hashed(
        {
            "schema": f"{acquisition.VERSION}_public_receipt",
            "version": acquisition.VERSION,
            "status": "formal_acquisition_complete",
            "design_sha256": acquisition.DESIGN_SHA256,
            "implementation_freeze_sha256": FREEZE_SHA256,
            "source_qualification_receipt": _source_qualification_receipt(),
            "selection_secret_commitment_sha256": (
                acquisition._selection_secret_commitment(SECRET)
            ),
            "selection_secret_persisted_publicly": False,
            "candidate_counts_by_family": {
                family: 60 for family in acquisition.FAMILIES
            },
            "typed_exclusion_counts": {},
            "block_counts": dict(acquisition.BLOCK_COUNTS),
            "per_family_quota": dict(acquisition.PER_FAMILY_QUOTA),
            "selected_question_count": sum(acquisition.BLOCK_COUNTS.values()),
            "selected_table_count": sum(acquisition.BLOCK_COUNTS.values()),
            "question_and_table_disjoint": True,
            "corpus_unit_count": acquisition.CORPUS_UNIT_COUNT,
            "corpus_unit_type_counts": {
                kind: type_counts[kind]
                for kind in ("table_row", "linked_passage")
            },
            "private_pack_file_sha256s": dict(sorted(file_hashes.items())),
            "F_search_label_pack_created": False,
            "raw_question_answer_table_or_unit_identity_persisted_publicly": False,
            "online_evaluator_calls": 0,
            "retry_replay_or_resample": 0,
        },
        "acquisition_receipt_sha256",
    )
    _write_private_json(acquisition_root / acquisition.PUBLIC_FILENAME, public)
    acquisition_root.chmod(0o500)
    boundary = controller.AcquisitionBoundary(
        tmp_path, expected_freeze_sha256=FREEZE_SHA256
    )
    try:
        yield tmp_path, boundary
    finally:
        boundary.close()
        acquisition_root.chmod(0o700)


def _feature_seal(view: controller.BlockView) -> runner.FeatureSeal:
    traces: list[evaluator.RecipeTrace] = []
    for item_i, item in enumerate(view.items):
        for recipe_i, recipe in enumerate(runner.RECIPE_IDS):
            traces.append(
                evaluator.RecipeTrace.from_mapping(
                    item_commitment_sha256=item.item_commitment_sha256,
                    recipe_id=recipe,
                    behavior_sha256=runner.stable_hash(
                        [item.item_commitment_sha256, recipe]
                    ),
                    features={
                        name: Fraction(item_i + recipe_i + feature_i, 10)
                        for feature_i, name in enumerate(runner.FEATURE_ORDER)
                    },
                )
            )
    return runner.seal_feature_matrix(block=view.block, traces=traces)


def _controller_root(project: Path) -> Path:
    root = project / controller.CONTROLLER_ROOT_RELATIVE
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _archive_capability(
    root: Path,
    *,
    block: str,
    feature_seal: runner.FeatureSeal,
    acquisition_receipt_sha256: str,
    corpus_pack_sha256: str,
    graph_sha256: str,
    block_view_sha256: str,
    durable: bool,
) -> controller.ArchiveCapability:
    commitments = feature_seal.item_commitments
    hippo_top5 = [[commitment, [0, 1, 2, 3, 4]] for commitment in commitments]
    execution = _self_hashed(
        {
            "schema": f"{controller.VERSION}_label_free_execution_receipt",
            "version": controller.VERSION,
            "block": block,
            "item_count": len(commitments),
            "acquisition_receipt_sha256": acquisition_receipt_sha256,
            "corpus_pack_sha256": corpus_pack_sha256,
            "block_view_sha256": block_view_sha256,
            "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
            "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
            "graph_sha256": graph_sha256,
            "minilm_index_sha256": "5" * 64,
            "hipporag_build_receipt_sha256": "6" * 64,
            "combined_hipporag_retrieval_receipt_sha256": "7" * 64,
            "combined_query_schedule_sha256": "8" * 64,
            "projected_hipporag_top5_sha256": controller.stable_hash(hippo_top5),
            "local_worker_cap": controller.LOCAL_WORKER_CAP,
            "all_item_matrices_eagerly_submitted_before_join": True,
            "local_and_official_jobs_submitted_before_join": True,
            "labels_family_gold_or_utility_accessed": False,
            "raw_query_or_corpus_text_persisted": False,
            "online_evaluator_calls": 0,
        },
        "execution_receipt_sha256",
    )
    traces_by_item = {
        commitment: [
            trace.payload()
            for trace in feature_seal.traces
            if trace.item_commitment_sha256 == commitment
        ]
        for commitment in commitments
    }
    items = []
    for commitment in commitments:
        feature_traces = traces_by_item[commitment]
        actions = [
            {
                "recipe_id": trace["recipe_id"],
                "output_top5": [0, 1, 2, 3, 4],
                "retained_raw_top3": [0, 1, 2],
                "selection_steps": [],
                "raw_dense_order_sha256": "9" * 64,
                "graph_sha256": graph_sha256,
                "query_sha256": "a" * 64,
                "semantic_tensor_sha256": "b" * 64,
                "reachability_sha256": "c" * 64,
                "candidate_scan_sha256": "d" * 64,
                "candidate_universe_size": 609,
                "candidate_score_evaluations": 0,
                "semantic_cell_scan_count": 0,
                "hipporag_candidate_or_feature_count": 0,
                "trace_sha256": trace["behavior_sha256"],
            }
            for trace in feature_traces
        ]
        items.append(
            {
                "item_commitment_sha256": commitment,
                "actions": actions,
                "feature_traces": feature_traces,
            }
        )
    archive = _self_hashed(
        {
            "schema": f"{controller.VERSION}_label_free_archive",
            "version": controller.VERSION,
            "block": block,
            "item_count": len(commitments),
            "acquisition_receipt_sha256": acquisition_receipt_sha256,
            "corpus_pack_sha256": corpus_pack_sha256,
            "block_view_sha256": block_view_sha256,
            "graph_sha256": graph_sha256,
            "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
            "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
            "execution_receipt": execution,
            "feature_receipt": feature_seal.receipt,
            "hipporag_top5": hippo_top5,
            "items": items,
            "complete_action_and_feature_traces_persisted": True,
            "raw_query_corpus_label_family_gold_or_utility_persisted": False,
            "online_evaluator_calls": 0,
        },
        "archive_sha256",
    )
    raw = _canonical_bytes(archive, newline=True)
    archive_file_sha256 = hashlib.sha256(raw).hexdigest()
    seal = _self_hashed(
        {
            "schema": f"{controller.VERSION}_label_free_archive_seal",
            "version": controller.VERSION,
            "block": block,
            "acquisition_receipt_sha256": acquisition_receipt_sha256,
            "corpus_pack_sha256": corpus_pack_sha256,
            "block_view_sha256": block_view_sha256,
            "graph_sha256": graph_sha256,
            "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
            "execution_receipt_sha256": execution["execution_receipt_sha256"],
            "archive_file_sha256": archive_file_sha256,
            "archive_sha256": archive["archive_sha256"],
            "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
            "label_pack_opened_before_seal": False,
        },
        "archive_seal_sha256",
    )
    capability = controller.ArchiveCapability(
        block=block,
        receipt_json=_canonical_bytes(seal).decode("ascii"),
    )
    if durable:
        _write_private_json(root / f"{block}.label_free_archive.json", archive)
        _write_private_json(root / f"{block}.archive_seal.json", seal)
    return capability


def _policy_matrix(count: int) -> tuple[evaluator.RecipeTrace, ...]:
    patterns = {
        "R0_DENSE5": (0, 0),
        "R1_P6_DIRECT_B2": (10, 0),
        "R2_P6_PATH1_B2": (0, 1),
        "R3_P6_PATH2_B2": (0, 2),
    }
    traces: list[evaluator.RecipeTrace] = []
    for item_i in range(count):
        commitment = runner.stable_hash(["policy", count, item_i])
        for recipe in runner.RECIPE_IDS:
            first, remainder = patterns[recipe]
            traces.append(
                evaluator.RecipeTrace.from_mapping(
                    item_commitment_sha256=commitment,
                    recipe_id=recipe,
                    behavior_sha256=runner.stable_hash([commitment, recipe]),
                    features={
                        name: Fraction(first if feature_i == 0 else remainder)
                        for feature_i, name in enumerate(runner.FEATURE_ORDER)
                    },
                )
            )
    return tuple(traces)


def _policy_seal() -> runner.PolicySeal:
    a_features = runner.seal_feature_matrix(
        block="A_form",
        traces=_policy_matrix(runner.BLOCK_COUNTS["A_form"]),
    )
    utilities = {
        (trace.item_commitment_sha256, trace.recipe_id): (
            2 if trace.recipe_id == "R1_P6_DIRECT_B2" else 0
        )
        for trace in a_features.traces
    }
    fit = runner.fit_e2(
        feature_seal=a_features,
        utilities=utilities,
        fold_secret=b"F" * 32,
    )
    f_features = runner.seal_feature_matrix(
        block="F_search",
        traces=_policy_matrix(runner.BLOCK_COUNTS["F_search"]),
    )
    return runner.freeze_f_policies(feature_seal=f_features, fit_seal=fit)


def _nonpromoted_a_hold_score() -> runner.AnchorScoreSeal:
    commitments = tuple(
        runner.stable_hash(["nonpromoted", index])
        for index in range(runner.BLOCK_COUNTS["A_hold"])
    )
    traces = tuple(
        evaluator.RecipeTrace.from_mapping(
            item_commitment_sha256=commitment,
            recipe_id=recipe,
            behavior_sha256=runner.stable_hash([commitment, recipe]),
            features={name: Fraction(recipe_i) for name in runner.FEATURE_ORDER},
        )
        for commitment in commitments
        for recipe_i, recipe in enumerate(runner.RECIPE_IDS)
    )
    features = runner.seal_feature_matrix(block="A_hold", traces=traces)
    hippo = runner.seal_hippo_retrievals(
        block="A_hold",
        rows=tuple(
            runner.HippoRetrieval(commitment, (0, 1, 2, 3, 4))
            for commitment in commitments
        ),
    )
    policies = _policy_seal()
    zero_test = runner._sign_flip_payload(
        (Fraction(0),) * runner.BLOCK_COUNTS["A_hold"]
    )
    body: dict[str, object] = {
        "schema": f"{runner.VERSION}_A_hold_score_receipt",
        "version": runner.VERSION,
        "block": "A_hold",
        "item_count": runner.BLOCK_COUNTS["A_hold"],
        "logical_RAW_HippoRAG_Agent_work_units": (
            3 * runner.BLOCK_COUNTS["A_hold"]
        ),
        "anchor_feature_receipt_sha256": features.feature_receipt_sha256,
        "policy_receipt_sha256": policies.policy_receipt_sha256,
        "hipporag_retrieval_matrix_sha256": hippo.retrieval_matrix_sha256,
        "item_commitment_set_sha256": features.item_commitment_set_sha256,
        "late_opened_label_matrix_sha256": runner.stable_hash(
            ["synthetic nonpromotion labels"]
        ),
        "A_hold_authorization_score_receipt_sha256": None,
        "E0_recipe_id": policies.e0_recipe_id,
        "E2_recipe_id": policies.e2_recipe_id,
        "evaluator_comparison_identifiable": policies.identifiable,
        "E2_minus_E0": zero_test,
        "E2_minus_HippoRAG": zero_test,
        "E2_minus_RAW": zero_test,
        "E2_minus_HippoRAG_family_sums": {
            family: [0, 1] for family in runner.FAMILIES
        },
        "family_item_counts": runner.BLOCK_FAMILY_COUNTS["A_hold"],
        "complete_counts": {
            "E0": 0,
            "E2": 0,
            "HippoRAG": 0,
            "RAW": 0,
        },
        "A_hold_real_domain_primary_passed": False,
        "evaluator_promoted": False,
        "M_L5_passed": None,
        "RAW_complete_advantage_overcome": False,
        "item_level_utility_values_persisted": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    receipt = _self_hashed(body, "score_receipt_sha256")
    return runner.AnchorScoreSeal(
        block="A_hold",
        anchor_features=features,
        hippo_retrievals=hippo,
        policies=policies,
        a_hold_authorization=None,
        receipt_json=_canonical_bytes(receipt).decode("ascii"),
    )


def test_controller_import_and_source_compile() -> None:
    imported = importlib.import_module(
        "assumption_agent.benchmarks.hybridqa_p6_e2_formal_controller_v1"
    )
    source_path = Path(imported.__file__ or "")
    compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
    assert imported.AcquisitionBoundary is controller.AcquisitionBoundary
    assert imported.BLOCK_COUNTS == acquisition.BLOCK_COUNTS


def test_acquisition_boundary_separates_public_receipt_and_private_views(
    synthetic_project: tuple[Path, controller.AcquisitionBoundary],
) -> None:
    project, boundary = synthetic_project
    public_text = json.dumps(boundary.public, sort_keys=True)
    assert boundary.public["selection_secret_persisted_publicly"] is False
    assert boundary.public["F_search_label_pack_created"] is False
    assert "Which synthetic value" not in public_text
    assert "-table-" not in public_text
    assert not (
        project
        / acquisition.ACQUISITION_RELATIVE
        / "F_search.labels.sealed.json"
    ).exists()
    view = boundary.load_view("A_form")
    assert len(view.items) == acquisition.BLOCK_COUNTS["A_form"]
    assert all(item.question.startswith("Which synthetic value") for item in view.items)
    assert all(
        (project / acquisition.ACQUISITION_RELATIVE / name).stat().st_mode
        & 0o077
        == 0
        for name in boundary.file_hashes
    )


def test_corpus_and_view_capabilities_are_instance_bound(
    synthetic_project: tuple[Path, controller.AcquisitionBoundary],
) -> None:
    project, boundary = synthetic_project
    corpus = boundary.load_corpus()
    view = boundary.load_view("A_form")
    root = _controller_root(project)
    with pytest.raises(controller.HybridQaFormalControllerError, match="view capability"):
        boundary.load_labels(
            "A_form",
            expected_view=replace(view, view_sha256="0" * 64),
            corpus=corpus,
            archive_capability=None,  # type: ignore[arg-type]
            feature_seal=None,  # type: ignore[arg-type]
            controller_root=root,
        )
    with pytest.raises(controller.HybridQaFormalControllerError, match="corpus capability"):
        boundary.load_labels(
            "A_form",
            expected_view=view,
            corpus=replace(corpus, pack_sha256="0" * 64),
            archive_capability=None,  # type: ignore[arg-type]
            feature_seal=None,  # type: ignore[arg-type]
            controller_root=root,
        )


def test_labels_cannot_open_from_an_in_memory_only_archive_capability(
    synthetic_project: tuple[Path, controller.AcquisitionBoundary],
) -> None:
    project, boundary = synthetic_project
    corpus = boundary.load_corpus()
    view = boundary.load_view("A_form")
    features = _feature_seal(view)
    root = _controller_root(project)
    capability = _archive_capability(
        root,
        block="A_form",
        feature_seal=features,
        acquisition_receipt_sha256=boundary.acquisition_receipt_sha256,
        corpus_pack_sha256=corpus.pack_sha256,
        graph_sha256=corpus.graph.graph_sha256,
        block_view_sha256=view.view_sha256,
        durable=False,
    )
    with pytest.raises(
        controller.HybridQaFormalControllerError, match="archive seal"
    ):
        boundary.load_labels(
            "A_form",
            expected_view=view,
            corpus=corpus,
            archive_capability=capability,
            feature_seal=features,
            controller_root=root,
        )


def test_m_search_view_rejects_missing_and_nonpromoted_authorization(
    synthetic_project: tuple[Path, controller.AcquisitionBoundary],
) -> None:
    project, boundary = synthetic_project
    root = _controller_root(project)
    with pytest.raises(controller.HybridQaFormalControllerError, match="lacks a promoted"):
        boundary.load_view("M_search")
    with pytest.raises(controller.HybridQaFormalControllerError, match="lacks a promoted"):
        boundary.load_view(
            "M_search",
            a_hold_authorization=_nonpromoted_a_hold_score(),
            controller_root=root,
        )


def test_durable_label_open_validates_all_three_gold_topologies(
    synthetic_project: tuple[Path, controller.AcquisitionBoundary],
) -> None:
    project, boundary = synthetic_project
    corpus = boundary.load_corpus()
    view = boundary.load_view("A_form")
    features = _feature_seal(view)
    root = _controller_root(project)
    capability = _archive_capability(
        root,
        block="A_form",
        feature_seal=features,
        acquisition_receipt_sha256=boundary.acquisition_receipt_sha256,
        corpus_pack_sha256=corpus.pack_sha256,
        graph_sha256=corpus.graph.graph_sha256,
        block_view_sha256=view.view_sha256,
        durable=True,
    )
    labels = boundary.load_labels(
        "A_form",
        expected_view=view,
        corpus=corpus,
        archive_capability=capability,
        feature_seal=features,
        controller_root=root,
    )
    assert Counter(row.family for row in labels.by_commitment.values()) == Counter(
        {family: acquisition.PER_FAMILY_QUOTA["A_form"] for family in acquisition.FAMILIES}
    )
    expected_shapes = {
        "TABLE_ONLY": (1, 0),
        "PASSAGE_ONLY": (1, 1),
        "DUAL_TABLE_PASSAGE": (2, 1),
    }
    representative: dict[str, controller.LabelRow] = {}
    for row in labels.by_commitment.values():
        units = tuple(corpus.graph.units[index] for index in row.gold_indices)
        observed = (
            sum(unit.unit_type == "table_row" for unit in units),
            sum(unit.unit_type == "linked_passage" for unit in units),
        )
        assert observed == expected_shapes[row.family]
        representative.setdefault(row.family, row)
    with pytest.raises(controller.HybridQaFormalControllerError, match="family topology"):
        passage = representative["PASSAGE_ONLY"]
        controller._validate_gold_topology(
            family="TABLE_ONLY",
            gold_indices=passage.gold_indices,
            graph=corpus.graph,
        )
