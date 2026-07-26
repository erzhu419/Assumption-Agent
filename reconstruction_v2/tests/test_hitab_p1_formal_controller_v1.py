from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np
import pytest

from assumption_agent.benchmarks import hitab_p1_dmc1_core_v1 as core
from assumption_agent.benchmarks import hitab_p1_formal_controller_v1 as formal
from assumption_agent.benchmarks import hitab_p1_runtime_v1 as runtime
from assumption_agent.benchmarks import hitab_p1_source_acquisition_v1 as source
from replication_runtime.birco_official_hipporag_v1 import contract as hippo_contract
from replication_runtime.bright_query_generator_v1 import contract as planner_contract


PERMUTATIONS = (
    tuple(range(10)),
    (3, 4, 5, 6, 7, 8, 9, 0, 1, 2),
    (6, 2, 7, 1, 5, 8, 0, 9, 4, 3),
    (7, 4, 1, 6, 0, 8, 5, 3, 9, 2),
)
UNIT_TYPES = (
    "DATA_CELL",
    "DATA_CELL",
    "DATA_CELL",
    "DATA_CELL",
    "DATA_CELL",
    "ROW_HEADER",
    "COLUMN_HEADER",
    "DERIVED_VALUE",
    "DATA_CELL",
    "DATA_CELL",
)
CE_Q6 = (
    (
        950_000,
        940_000,
        930_000,
        920_000,
        910_000,
        320_000,
        300_000,
        280_000,
        20_000,
        10_000,
    ),
    (50_000,) * 10,
    (
        160_000,
        150_000,
        140_000,
        130_000,
        120_000,
        850_000,
        820_000,
        790_000,
        30_000,
        20_000,
    ),
    (
        110_000,
        100_000,
        90_000,
        80_000,
        70_000,
        240_000,
        860_000,
        810_000,
        10_000,
        30_000,
    ),
    (40_000,) * 10,
)


def _logical_ordinal(text: str) -> int:
    match = re.search(r"token=L([0-9])", text)
    assert match is not None
    return int(match.group(1))


class _Planner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, canonical_input: bytes) -> bytes:
        item = planner_contract.parse_input(canonical_input)[0]
        self.calls += 1
        completion = json.dumps(
            {
                "constraint_query": "constraint facet",
                "entity_query": "neutral facet",
                "mechanism_query": "mechanism facet",
                "relation_query": "relation facet",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        output = planner_contract.build_output_item(
            ordinal=0,
            completion=completion,
            completion_token_count=16,
            query=item.query,
        )
        return planner_contract.canonical_json_bytes(
            planner_contract.output_payload((output,))
        )


class _Scorer:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, pairs):
        if self.fail:
            raise RuntimeError("synthetic scorer failure with private-like text")
        rows = tuple(pairs)
        self.calls += 1
        output = []
        facet_by_query = {
            "neutral facet": 1,
            "relation facet": 2,
            "mechanism facet": 3,
            "constraint facet": 4,
        }
        for query, unit in rows:
            facet = facet_by_query.get(query, 0)
            probability = CE_Q6[facet][_logical_ordinal(unit)] / 1_000_000
            output.append(math.log(probability / (1.0 - probability)))
        return tuple(output)


def _unit_vector(logical: int) -> np.ndarray:
    row = np.zeros(runtime.EMBEDDING_DIMENSION, dtype=np.float32)
    if logical < 5:
        row[0] = np.float32(math.sqrt(0.18))
        row[1] = np.float32(math.sqrt(0.76))
        row[10 + logical] = np.float32(math.sqrt(0.06))
    else:
        row[0] = np.float32(math.sqrt(0.18))
        row[20 + logical] = np.float32(math.sqrt(0.82))
    row /= np.float32(np.linalg.norm(row.astype(np.float64)))
    return row


class _Encoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        rows = []
        for text in texts:
            match = re.search(r"token=L([0-9])", text)
            if match is not None:
                row = _unit_vector(int(match.group(1)))
            else:
                row = np.zeros(
                    runtime.EMBEDDING_DIMENSION, dtype=np.float32
                )
                if "synthetic formal question" in text:
                    row[1] = 1.0
                elif text == "neutral facet":
                    row[40] = 1.0
                elif text == "relation facet":
                    row[25] = 1.0
                elif text == "mechanism facet":
                    row[26] = 0.8
                    row[27] = 0.6
                else:
                    row[41] = 1.0
                row /= np.float32(np.linalg.norm(row.astype(np.float64)))
            rows.append(row)
        return np.stack(rows).astype(np.float32)


class _CacheReleaser:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return formal.self_hashed(
            {
                "model_offload_or_reload": False,
                "physical_gpu": 0,
                "schema": "hitab_p1_gpu0_unused_cuda_cache_release_v1",
                "study_id": formal.STUDY_ID,
                "torch_cuda_empty_cache_called": True,
            }
        )


class _Hippo:
    def __init__(self, *, perfect_proof: bool = False) -> None:
        self.calls: list[tuple[int, int]] = []
        self.formation_call_counts: list[tuple[int, int]] = []
        self.scorer: _Scorer | None = None
        self.cache_release_calls = 0
        self.perfect_proof = perfect_proof
        self._barrier = threading.Barrier(2, timeout=10)
        self._lock = threading.Lock()
        self._active_total = 0
        self._active_by_gpu = {0: 0, 1: 0}
        self.maximum_active_total = 0
        self.maximum_active_by_gpu = {0: 0, 1: 0}

    def __call__(
        self,
        canonical_input: bytes,
        *,
        physical_gpu: int,
        cpu_thread_limit: int,
        launch_ack,
    ) -> bytes:
        value = json.loads(canonical_input.decode("ascii"))
        checked = hippo_contract.validate_input(
            value["work_id"],
            value["objective"],
            value["query"],
            value["documents"],
            value["common_projection_sha256"],
        )
        with self._lock:
            self.calls.append((physical_gpu, cpu_thread_limit))
            self.formation_call_counts.append(
                (
                    physical_gpu,
                    self.scorer.calls if self.scorer is not None else -1,
                )
            )
            self._active_total += 1
            self._active_by_gpu[physical_gpu] += 1
            self.maximum_active_total = max(
                self.maximum_active_total, self._active_total
            )
            self.maximum_active_by_gpu[physical_gpu] = max(
                self.maximum_active_by_gpu[physical_gpu],
                self._active_by_gpu[physical_gpu],
            )
        try:
            launch_ack()
            self._barrier.wait()
            documents = checked[3]
            priority = (
                (0, 5, 6, 7, 1, 2, 3, 4, 8, 9)
                if self.perfect_proof
                else tuple(range(10))
            )
            priority_index = {
                logical: index for index, logical in enumerate(priority)
            }
            ranking = tuple(
                row.ordinal
                for row in sorted(
                    documents,
                    key=lambda row: priority_index[
                        _logical_ordinal(row.text)
                    ],
                )
            )
            payload = hippo_contract.output_payload(
                work_id=checked[0],
                common_projection_sha256=checked[4],
                candidate_count=len(documents),
                rank_ordinals=ranking,
                graph_nodes=10,
                graph_edges=9,
            )
            return hippo_contract.canonical_json_bytes(payload)
        finally:
            with self._lock:
                self._active_total -= 1
                self._active_by_gpu[physical_gpu] -= 1


def _item(
    *,
    block: str,
    index: int,
) -> tuple[formal.FormalItemView, tuple[int, ...]]:
    permutation = PERMUTATIONS[index % len(PERMUTATIONS)]
    units = tuple(
        (
            f"VALUE type=synthetic surface=UNIT_SECRET_{logical} "
            f"token=L{logical} cohort={block}_{index} | "
            "LEFT_PATH synthetic | TOP_PATH fixture"
        )
        for logical in permutation
    )
    inverse = {logical: local for local, logical in enumerate(permutation)}
    edges = tuple(
        sorted(
            core.TypedEdge(
                min(inverse[left], inverse[right]),
                max(inverse[left], inverse[right]),
                core.SOURCE_NATIVE_EDGE_TYPE,
            )
            for left, right in ((0, 5), (5, 6), (6, 7))
        )
    )
    runtime_item = runtime.RuntimeItem(
        question=(
            f"synthetic formal question QUESTION_SECRET_{block}_{index}"
        ),
        ordered_unit_strings=units,
        corpus_commitment=runtime.ordered_corpus_commitment(units),
        unit_types=tuple(UNIT_TYPES[logical] for logical in permutation),
        typed_edges=edges,
    )
    work_id = "hitab-work-v1-" + formal.stable_hash(
        {"block": block, "index": index, "schema": "synthetic-work-v1"}
    )
    return formal.FormalItemView(work_id, runtime_item), permutation


class _Boundary:
    def __init__(
        self,
        root: Path,
        *,
        positive: bool,
        invalid_aform_qrel: bool = False,
    ) -> None:
        self.root = root
        self.positive = positive
        self.invalid_aform_qrel = invalid_aform_qrel
        self.events: list[str] = []
        self.hippo: _Hippo | None = None
        self._permutations: dict[str, tuple[int, ...]] = {}
        self.blocks = {}
        for block, count in formal.BLOCK_COUNTS.items():
            items = []
            for index in range(count):
                item, permutation = _item(block=block, index=index)
                items.append(item)
                self._permutations[item.work_id] = permutation
            self.blocks[block] = formal.BlockView.create(block, items)
        self.claim = formal.AcquisitionClaim.create(
            source_identity_commitment="a" * 64,
            initial_selection_commitment="b" * 64,
        )

    def claim_formal_attempt(
        self, formal_marker_sha256: str
    ) -> formal.AcquisitionClaim:
        marker_path = self.root / formal.FORMAL_MARKER_FILENAME
        assert marker_path.is_file()
        marker = json.loads(marker_path.read_text("ascii"))
        assert marker["self_sha256"] == formal_marker_sha256
        self.events.append("claim")
        return self.claim

    def load_label_free_block(
        self,
        block: str,
        authorization=None,
    ) -> formal.BlockView:
        if block == "M_search":
            assert authorization is not None
            body = dict(authorization)
            claimed = body.pop("self_sha256")
            assert formal.stable_hash(body) == claimed
            assert (
                authorization[
                    "aggregate_exact_utility_net_strictly_positive"
                ]
                is True
            )
            assert (
                self.root / formal.PROMOTION_AUTHORIZATION_FILENAME
            ).is_file()
        else:
            assert authorization is None
        self.events.append(f"load:{block}")
        return self.blocks[block]

    def release_qrels_after_action_seal(
        self,
        block: str,
        custody_path: Path,
        sealed_action_archive,
    ) -> formal.QrelPack:
        path = self.root / f"{block}.actions.private.json"
        assert custody_path == path
        assert path.is_file()
        assert (path.stat().st_mode & 0o777) == 0o400
        action = json.loads(path.read_text("ascii"))
        assert action == sealed_action_archive
        action_archive_sha256 = action["self_sha256"]
        assert action["item_count"] == formal.BLOCK_COUNTS[block]
        assert len(action["records"]) == formal.BLOCK_COUNTS[block]
        assert {row["work_id"] for row in action["records"]} == {
            row.work_id for row in self.blocks[block].items
        }
        if block == "A_form":
            assert action["registry_stage_complete"] is True
            assert all("registry" in row for row in action["records"])
        else:
            assert self.hippo is not None
            expected_hippo_calls = (
                formal.BLOCK_COUNTS["A_hold"]
                + (
                    formal.BLOCK_COUNTS["M_search"]
                    if block == "M_search"
                    else 0
                )
            )
            assert len(self.hippo.calls) == expected_hippo_calls
            assert re.fullmatch(
                r"[0-9a-f]{64}", action["e1_model_sha256"]
            )
            cache_receipt = action[
                "gpu0_unused_cuda_cache_release_receipt"
            ]
            cache_body = dict(cache_receipt)
            cache_claim = cache_body.pop("self_sha256")
            assert formal.stable_hash(cache_body) == cache_claim
            assert cache_receipt["torch_cuda_empty_cache_called"] is True
            assert cache_receipt["model_offload_or_reload"] is False
            for record in action["records"]:
                assert set(record["arms"]) == {
                    "E0",
                    "E1",
                    "HippoRAG",
                    "RAW",
                }
                assert len(
                    {
                        arm["corpus_commitment"]
                        for arm in record["arms"].values()
                    }
                ) == 1
        self.events.append(f"release:{block}")
        rows = []
        for index, item in enumerate(self.blocks[block].items):
            permutation = self._permutations[item.work_id]
            inverse = {
                logical: local
                for local, logical in enumerate(permutation)
            }
            if self.positive:
                selected = (0, 5, 6, 7)
            else:
                selected = (0,)
            proof = core.ProofDNF(
                alternatives=(
                    tuple((inverse[logical],) for logical in selected),
                ),
                corpus_commitment=item.runtime_item.corpus_commitment,
            )
            if self.invalid_aform_qrel and block == "A_form":
                proof = core.ProofDNF(
                    alternatives=(((inverse[0], inverse[1]),),),
                    corpus_commitment=item.runtime_item.corpus_commitment,
                )
            rows.append(
                formal.QrelRow(
                    work_id=item.work_id,
                    family=formal.FAMILIES[index % 3],
                    proof=proof,
                    corpus_commitment=(
                        item.runtime_item.corpus_commitment
                    ),
                    qrel_ordinal_mapping_commitment=(
                        proof.ordinal_mapping_commitment
                    ),
                )
            )
        return formal.QrelPack.create(
            block=block,
            action_archive_sha256=action_archive_sha256,
            rows=rows,
        )


def _run(
    root: Path,
    boundary: _Boundary,
    *,
    scorer: _Scorer | None = None,
    hippo: _Hippo | None = None,
) -> tuple[dict[str, object], _Planner, _Scorer, _Encoder, _Hippo]:
    planner = _Planner()
    actual_scorer = scorer or _Scorer()
    encoder = _Encoder()
    actual_hippo = hippo or _Hippo()
    cache_releaser = _CacheReleaser()
    actual_hippo.scorer = actual_scorer
    boundary.hippo = actual_hippo
    terminal = formal.run_formal_controller(
        work_root=root,
        execution_binding_sha256="c" * 64,
        acquisition=boundary,
        planner_runner=planner,
        cross_encoder_scorer=actual_scorer,
        minilm_encoder=encoder,
        hippo_runner=actual_hippo,
        gpu0_cache_releaser=cache_releaser,
    )
    actual_hippo.cache_release_calls = cache_releaser.calls
    return dict(terminal), planner, actual_scorer, encoder, actual_hippo


def test_positive_lifecycle_promotes_then_measures_m_and_stays_private(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    boundary = _Boundary(root, positive=True)
    terminal, planner, scorer, encoder, hippo = _run(root, boundary)

    assert terminal["status"] == (
        "terminal_complete_after_A_hold_promotion_and_M_search"
    )
    assert terminal["A_hold"]["promotion_passed"] is True
    assert terminal["A_hold"]["reality_primary_passed"] is True
    assert terminal["M_search"]["opened_after_promotion"] is True
    assert terminal["M_search"]["L5_passed"] is True
    assert boundary.events == [
        "claim",
        "load:A_form",
        "release:A_form",
        "load:A_hold",
        "release:A_hold",
        "load:M_search",
        "release:M_search",
    ]
    expected_compiles = (
        formal.BLOCK_COUNTS["A_form"]
        + formal.BLOCK_COUNTS["A_hold"]
        + formal.BLOCK_COUNTS["M_search"]
    )
    assert planner.calls == scorer.calls == encoder.calls == expected_compiles
    assert len(hippo.calls) == 72
    assert [row[0] for row in hippo.calls].count(0) == 36
    assert [row[0] for row in hippo.calls].count(1) == 36
    assert hippo.maximum_active_total == 2
    assert hippo.maximum_active_by_gpu == {0: 1, 1: 1}
    assert hippo.cache_release_calls == 2
    assert all(row[1] == 4 for row in hippo.calls)
    ahold_starts = hippo.formation_call_counts[:36]
    m_starts = hippo.formation_call_counts[36:]
    assert next(count for gpu, count in ahold_starts if gpu == 1) < 144
    assert next(count for gpu, count in ahold_starts if gpu == 0) >= 144
    assert next(count for gpu, count in m_starts if gpu == 1) < 180
    assert next(count for gpu, count in m_starts if gpu == 0) >= 180

    terminal_text = json.dumps(terminal, sort_keys=True)
    assert "QUESTION_SECRET" not in terminal_text
    assert "UNIT_SECRET" not in terminal_text
    assert '"work_id"' not in terminal_text
    assert '"rows"' not in terminal_text
    assert (root.stat().st_mode & 0o777) == 0o700
    assert all(
        (path.stat().st_mode & 0o777) == 0o400
        for path in root.iterdir()
        if path.is_file()
    )
    aform_text = (root / "A_form.actions.private.json").read_text(
        "ascii"
    ).casefold()
    assert all(
        token not in aform_text
        for token in ('"qrel"', '"proof"', '"gold"', '"family"')
    )
    model_archive = json.loads(
        (root / "E1.model.private.json").read_text("ascii")
    )
    assert model_archive["formation_item_count"] == 108
    assert len(model_archive["formation_bindings"]) == 108
    assert model_archive["formation_binding_set_commitment"] == (
        formal.stable_hash(model_archive["formation_bindings"])
    )
    ahold_action = json.loads(
        (root / "A_hold.actions.private.json").read_text("ascii")
    )
    m_action = json.loads(
        (root / "M_search.actions.private.json").read_text("ascii")
    )
    assert (
        ahold_action["e1_model_sha256"]
        == m_action["e1_model_sha256"]
        == model_archive["self_sha256"]
    )


def test_valid_nonpromotion_leaves_m_unopened_and_reality_does_not_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    boundary = _Boundary(root, positive=False)
    terminal, _planner, _scorer, _encoder, hippo = _run(root, boundary)

    assert terminal["status"] == (
        "terminal_A_hold_E1_not_promoted_M_search_unopened"
    )
    assert terminal["A_hold"]["promotion_passed"] is False
    assert terminal["M_search"]["opened_after_promotion"] is False
    assert terminal["M_search"]["L5_passed"] is None
    assert boundary.events == [
        "claim",
        "load:A_form",
        "release:A_form",
        "load:A_hold",
        "release:A_hold",
    ]
    assert not (root / formal.PROMOTION_AUTHORIZATION_FILENAME).exists()
    assert not (root / "M_search.actions.private.json").exists()
    assert len(hippo.calls) == formal.BLOCK_COUNTS["A_hold"]
    assert hippo.cache_release_calls == 1


def test_reality_failure_does_not_block_promoted_m_search(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    boundary = _Boundary(root, positive=True)
    terminal, *_rest = _run(
        root, boundary, hippo=_Hippo(perfect_proof=True)
    )
    assert terminal["A_hold"]["promotion_passed"] is True
    assert terminal["A_hold"]["reality_primary_passed"] is False
    assert terminal["M_search"]["opened_after_promotion"] is True
    assert "load:M_search" in boundary.events
    assert boundary.events.count("load:M_search") == 1


def test_failure_is_safe_terminal_and_same_root_cannot_replay(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    boundary = _Boundary(root, positive=True)
    with pytest.raises(
        formal.HitabP1FormalControllerError, match="failed closed"
    ):
        _run(root, boundary, scorer=_Scorer(fail=True))
    terminal_path = root / formal.FORMAL_TERMINAL_FILENAME
    terminal_raw = terminal_path.read_bytes()
    terminal = json.loads(terminal_raw)
    assert terminal["status"] == "terminal_formal_failure_no_retry"
    assert "synthetic scorer failure" not in terminal_raw.decode("ascii")
    assert boundary.events == ["claim", "load:A_form"]
    assert not any(event.startswith("release:") for event in boundary.events)

    with pytest.raises(
        formal.HitabP1FormalControllerError, match="already exists"
    ):
        _run(root, boundary)
    assert terminal_path.read_bytes() == terminal_raw
    assert boundary.events == ["claim", "load:A_form"]


def test_pre_ack_hippo_failure_wakes_and_never_forms_ahold(
    tmp_path: Path,
) -> None:
    class PreAckHippo(_Hippo):
        def __call__(self, _canonical_input: bytes, **_kwargs) -> bytes:
            raise RuntimeError("synthetic pre-ack Hippo failure")

    root = tmp_path / "formal"
    boundary = _Boundary(root, positive=True)
    planner = _Planner()
    scorer = _Scorer()
    encoder = _Encoder()
    hippo = PreAckHippo()
    cache = _CacheReleaser()
    hippo.scorer = scorer
    boundary.hippo = hippo
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            formal.run_formal_controller,
            work_root=root,
            execution_binding_sha256="c" * 64,
            acquisition=boundary,
            planner_runner=planner,
            cross_encoder_scorer=scorer,
            minilm_encoder=encoder,
            hippo_runner=hippo,
            gpu0_cache_releaser=cache,
        )
        with pytest.raises(
            formal.HitabP1FormalControllerError,
            match="launch acknowledgement",
        ):
            future.result(timeout=30)
    assert planner.calls == formal.BLOCK_COUNTS["A_form"]
    assert cache.calls == 0
    assert boundary.events == [
        "claim",
        "load:A_form",
        "release:A_form",
        "load:A_hold",
    ]
    terminal = json.loads(
        (root / formal.FORMAL_TERMINAL_FILENAME).read_text("ascii")
    )
    assert terminal["status"] == "terminal_formal_failure_no_retry"


def test_merged_same_literal_bucket_is_rejected_after_aform_seal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    boundary = _Boundary(
        root, positive=True, invalid_aform_qrel=True
    )
    with pytest.raises(
        formal.HitabP1FormalControllerError,
        match="one alternative of singleton buckets",
    ):
        _run(root, boundary)
    assert (root / "A_form.actions.private.json").is_file()
    assert not (root / "E1.model.private.json").exists()
    assert boundary.events == [
        "claim",
        "load:A_form",
        "release:A_form",
    ]


def test_production_acquisition_boundary_and_controller_integrate_without_m(
    tmp_path: Path,
) -> None:
    formal_root = tmp_path / "formal"
    control_root = tmp_path / "acquisition"
    control_root.mkdir(mode=0o700)
    synthetic = _Boundary(formal_root, positive=False)
    bridge_views = {}
    for block in ("A_form", "A_hold"):
        formal_view = synthetic.blocks[block]
        bridge_items = tuple(
            source.BridgeViewItem(
                work_id=row.work_id,
                question=row.runtime_item.question,
                ordered_unit_strings=(
                    row.runtime_item.ordered_unit_strings
                ),
                unit_types=row.runtime_item.unit_types,
                typed_edges=tuple(
                    source.TableTypedEdge(
                        edge.source_ordinal,
                        edge.target_ordinal,
                        edge.edge_type,
                    )
                    for edge in row.runtime_item.typed_edges
                ),
                corpus_commitment=(
                    row.runtime_item.corpus_commitment
                ),
            )
            for row in formal_view.items
        )
        bridge_view = source.BridgeBlockView(
            block=block,
            items=bridge_items,
            view_sha256=formal_view.view_sha256,
        )
        bridge_views[block] = bridge_view
        source._persist_block_view(
            control_root / source.BLOCK_VIEW_FILENAMES[block],
            bridge_view,
        )
        qrel_rows = []
        for index, row in enumerate(bridge_items):
            logical_zero = next(
                ordinal
                for ordinal, text in enumerate(
                    row.ordered_unit_strings
                )
                if _logical_ordinal(text) == 0
            )
            proof = core.ProofDNF(
                alternatives=(((logical_zero,),),),
                corpus_commitment=row.corpus_commitment,
            )
            qrel_rows.append(
                source.BridgeQrelRow(
                    work_id=row.work_id,
                    family=formal.FAMILIES[index % 3],
                    qrel=proof,
                    corpus_commitment=row.corpus_commitment,
                    qrel_ordinal_mapping_commitment=(
                        proof.ordinal_mapping_commitment
                    ),
                )
            )
        source._persist_qrel_custody(
            control_root / source.QREL_CUSTODY_FILENAMES[block],
            block=block,
            block_view_sha256=bridge_view.view_sha256,
            rows=tuple(qrel_rows),
        )

    source_commitment = "a" * 64
    selection_commitment = "b" * 64
    initial_run = source.InitialSelectionRun(
        block_views=bridge_views,
        safe_receipt=source.self_hashed(
            {
                "block_view_sha256": {
                    block: view.view_sha256
                    for block, view in bridge_views.items()
                },
                "schema": "synthetic_initial_selection_receipt_v1",
                "selection_commitment": selection_commitment,
                "source_identity_commitment": source_commitment,
                "study_id": formal.STUDY_ID,
            }
        ),
    )
    identities = {
        key: source.VerifiedFileIdentity(
            key=key,
            size_bytes=1,
            sha256=str(index + 1) * 64,
            git_blob_sha1=str(index + 1) * 40,
            raw_newline_count=1 if key != "TABLES" else None,
        )
        for index, key in enumerate(("TRAIN", "DEV", "TEST", "TABLES"))
    }
    verified = source.VerifiedSourceSet(
        identities=identities,
        safe_receipt={
            "source_identity_commitment": source_commitment
        },
    )
    # None of these deliberately absent paths may be opened in this valid
    # nonpromotion lifecycle; in particular TEST remains untouched.
    source_paths = {
        key: tmp_path / "absent-source" / key
        for key in ("TRAIN", "DEV", "TEST", "TABLES")
    }
    production_boundary = source.ProductionFormalAcquisitionBoundary(
        source_paths=source_paths,
        verified_sources=verified,
        control_root=control_root,
        formal_work_root=formal_root,
        initial_run=initial_run,
        quota_per_family={
            "A_form": 36,
            "A_hold": 12,
            "M_search": 12,
        },
        public_exposure_hashes={
            "item_id": frozenset(),
            "question": frozenset(),
            "table_id": frozenset(),
        },
    )
    planner = _Planner()
    scorer = _Scorer()
    encoder = _Encoder()
    hippo = _Hippo()
    cache = _CacheReleaser()
    hippo.scorer = scorer
    terminal = formal.run_formal_controller(
        work_root=formal_root,
        execution_binding_sha256="c" * 64,
        acquisition=production_boundary,
        planner_runner=planner,
        cross_encoder_scorer=scorer,
        minilm_encoder=encoder,
        hippo_runner=hippo,
        gpu0_cache_releaser=cache,
    )
    assert terminal["status"] == (
        "terminal_A_hold_E1_not_promoted_M_search_unopened"
    )
    assert not source_paths["TEST"].exists()
    assert not (formal_root / "M_search.actions.private.json").exists()
    assert (
        control_root / source.QREL_RELEASE_MARKER_FILENAMES["A_form"]
    ).is_file()
    assert (
        control_root / source.QREL_RELEASE_MARKER_FILENAMES["A_hold"]
    ).is_file()
    assert not (
        control_root / source.QREL_RELEASE_MARKER_FILENAMES["M_search"]
    ).exists()
    assert (
        control_root / source.FORMAL_CLAIM_FILENAME
    ).is_file()


def test_production_boundary_rejects_bad_action_before_any_qrel_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal_root = tmp_path / "formal"
    control_root = tmp_path / "acquisition"
    formal_root.mkdir(mode=0o700)
    control_root.mkdir(mode=0o700)
    bridge_views: dict[str, source.BridgeBlockView] = {}
    for block in ("A_form", "A_hold"):
        formal_item, _ = _item(block=block, index=0)
        runtime_item = formal_item.runtime_item
        bridge_item = source.BridgeViewItem(
            work_id=formal_item.work_id,
            question=runtime_item.question,
            ordered_unit_strings=runtime_item.ordered_unit_strings,
            unit_types=runtime_item.unit_types,
            typed_edges=tuple(
                source.TableTypedEdge(
                    edge.source_ordinal,
                    edge.target_ordinal,
                    edge.edge_type,
                )
                for edge in runtime_item.typed_edges
            ),
            corpus_commitment=runtime_item.corpus_commitment,
        )
        payload = {
            "block": block,
            "items": [bridge_item.private_payload()],
        }
        view = source.BridgeBlockView(
            block=block,
            items=(bridge_item,),
            view_sha256=source.stable_hash(payload),
        )
        bridge_views[block] = view
        source._persist_block_view(
            control_root / source.BLOCK_VIEW_FILENAMES[block],
            view,
        )

    source_commitment = "a" * 64
    initial_run = source.InitialSelectionRun(
        block_views=bridge_views,
        safe_receipt=source.self_hashed(
            {
                "block_view_sha256": {
                    block: view.view_sha256
                    for block, view in bridge_views.items()
                },
                "schema": "synthetic_initial_selection_receipt_v1",
                "selection_commitment": "b" * 64,
                "source_identity_commitment": source_commitment,
                "study_id": formal.STUDY_ID,
            }
        ),
    )
    identities = {
        key: source.VerifiedFileIdentity(
            key=key,
            size_bytes=1,
            sha256=str(index + 1) * 64,
            git_blob_sha1=str(index + 1) * 40,
            raw_newline_count=1 if key != "TABLES" else None,
        )
        for index, key in enumerate(("TRAIN", "DEV", "TEST", "TABLES"))
    }
    verified = source.VerifiedSourceSet(
        identities=identities,
        safe_receipt={
            "source_identity_commitment": source_commitment,
        },
    )
    boundary = source.ProductionFormalAcquisitionBoundary(
        source_paths={
            key: tmp_path / "absent-source" / key
            for key in ("TRAIN", "DEV", "TEST", "TABLES")
        },
        verified_sources=verified,
        control_root=control_root,
        formal_work_root=formal_root,
        initial_run=initial_run,
        quota_per_family={
            "A_form": 1,
            "A_hold": 1,
            "M_search": 1,
        },
        public_exposure_hashes={
            "item_id": frozenset(),
            "question": frozenset(),
            "table_id": frozenset(),
        },
    )
    marker = formal.self_hashed(
        {
            "execution_binding_sha256": "c" * 64,
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": f"{formal.VERSION}_one_shot_marker_v1",
            "study_id": formal.STUDY_ID,
        }
    )
    formal._exclusive_bytes(
        formal_root / formal.FORMAL_MARKER_FILENAME,
        formal.canonical_bytes(marker),
    )
    boundary.claim_formal_attempt(str(marker["self_sha256"]))
    boundary.load_label_free_block("A_hold", None)

    malformed = formal.self_hashed(
        {
            "block": "A_hold",
            "block_view_sha256": bridge_views["A_hold"].view_sha256,
            "e1_model_sha256": "d" * 64,
            "four_arm_corpus_commitment_exact": True,
            "hipporag_queue_joined_before_archive": True,
            "item_count": 1,
            "records": [],
            "schema": (
                "hitab_p1_formal_controller_v1_A_hold_"
                "four_arm_action_archive_v1"
            ),
            "study_id": formal.STUDY_ID,
        }
    )
    action_path = formal_root / "A_hold.actions.private.json"
    formal._exclusive_bytes(
        action_path,
        formal.canonical_bytes(malformed),
    )
    qrel_read_count = 0
    original_loader = source._load_qrel_custody

    def tracking_loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal qrel_read_count
        qrel_read_count += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(source, "_load_qrel_custody", tracking_loader)
    with pytest.raises(source.HitabP1SourceError, match="header drifted"):
        boundary.release_qrels_after_action_seal(
            "A_hold",
            action_path,
            malformed,
        )
    assert qrel_read_count == 0
    assert not (
        control_root / source.QREL_RELEASE_MARKER_FILENAMES["A_hold"]
    ).exists()
    assert not (
        control_root / source.QREL_CUSTODY_FILENAMES["A_hold"]
    ).exists()


def test_public_controller_surface_is_source_free_and_label_free() -> None:
    assert set(formal.FormalItemView.__dataclass_fields__) == {
        "work_id",
        "runtime_item",
    }
    assert set(formal.BlockView.__dataclass_fields__) == {
        "block",
        "items",
        "view_sha256",
    }
    source = Path(formal.__file__).read_text("utf-8")
    assert "hitab_p1_source_acquisition_v1" not in source
    assert "source_paths" not in source
    assert "Ruoli" not in source
    assert "requests." not in source
