from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    bioasq_p1_formal_controller_v1 as ctl,
)
from assumption_agent.benchmarks import bioasq_p1_formal_source_v2 as source
from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core
from replication_runtime.bioasq_p1_formal_v1 import acquisition
from replication_runtime.bioasq_p1_formal_v1.contract import (
    BioasqP1FormalRuntimeError,
    canonical_bytes,
    stable_hash,
)


@pytest.fixture
def posix_tmp() -> Iterator[Path]:
    root = Path(
        tempfile.mkdtemp(prefix="bioasq-acquisition-", dir="/tmp")
    ).absolute()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@dataclass(frozen=True)
class _Fixture:
    outputs: source.FormalOutputPaths
    receipt: Mapping[str, object]
    controller_root: Path
    work_ids: Mapping[str, tuple[str, ...]]


class _HippoStub:
    def __init__(self) -> None:
        self.build_call_count = 0
        self.corpus: ctl.CorpusView | None = None

    def start_build(self, corpus: ctl.CorpusView) -> None:
        self.build_call_count += 1
        if self.build_call_count != 1:
            raise AssertionError("Hippo build started twice")
        self.corpus = corpus


def _write_bytes(path: Path, raw: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(mode)
    assert path.stat().st_nlink == 1
    assert stat.S_IMODE(path.stat().st_mode) == mode


def _write_source_json(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int,
) -> bytes:
    raw = source.canonical_bytes(value, newline=True)
    _write_bytes(path, raw, mode=mode)
    return raw


def _write_controller_json(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int = 0o400,
) -> bytes:
    raw = ctl.canonical_bytes(value)
    _write_bytes(path, raw, mode=mode)
    return raw


def _binding(
    raw: bytes,
    value: Mapping[str, object],
    *,
    mode: int,
    row_count: int,
) -> dict[str, object]:
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "mode": f"{mode:04o}",
        "row_count": row_count,
        "self_sha256": value["self_sha256"],
        "size_bytes": len(raw),
    }


def _work_id(block: str, ordinal: int) -> str:
    digest = hashlib.sha256(
        f"{block}:{ordinal}".encode("ascii")
    ).hexdigest()
    return "bioasq-work-v2-" + digest


def _fixture(root: Path) -> _Fixture:
    source_root = root / "source"
    controller_root = root / "controller"
    source_root.mkdir(mode=0o700)
    controller_root.mkdir(mode=0o700)
    outputs = source.FormalOutputPaths(
        private_selection_secret=source_root / "selection.secret.bin",
        public_corpus=source_root / "corpus.public.json",
        public_a_form=source_root / "A_form.public.json",
        public_f_search=source_root / "F_search.public.json",
        public_a_hold=source_root / "A_hold.public.json",
        public_m_search=source_root / "M_search.public.sealed.json",
        private_a_form_qrels=source_root / "A_form.qrels.private.json",
        private_a_hold_qrels=source_root / "A_hold.qrels.private.json",
        private_m_search_qrels=(
            source_root / "M_search.qrels.private.sealed.json"
        ),
        safe_selection_receipt=source_root / "selection.safe.json",
    )
    secret = b"S" * source.HMAC_SECRET_BYTES
    _write_bytes(
        outputs.private_selection_secret,
        secret,
        mode=0o600,
    )

    passages = [
        core.passage_public_payload(
            core.Passage(
                ordinal=ordinal,
                text=f"Synthetic evidence passage {ordinal}.",
            )
        )
        for ordinal in range(ctl.CORPUS_SIZE)
    ]
    corpus = source.self_hashed(
        {
            "passages": passages,
            "schema": source.PUBLIC_CORPUS_SCHEMA,
            "study_id": ctl.STUDY_ID,
            "version": source.VERSION,
        }
    )
    corpus_raw = _write_source_json(
        outputs.public_corpus,
        corpus,
        mode=0o600,
    )

    work_ids: dict[str, tuple[str, ...]] = {}
    public_values: dict[str, Mapping[str, object]] = {}
    public_raw: dict[str, bytes] = {}
    qrel_values: dict[str, Mapping[str, object]] = {}
    qrel_raw: dict[str, bytes] = {}
    global_ordinal = 0
    for block in source.BLOCKS:
        count = ctl.BLOCK_COUNTS[block]
        block_work_ids = tuple(_work_id(block, index) for index in range(count))
        work_ids[block] = block_work_ids
        items = [
            {
                "query_text": (
                    f"Synthetic biomedical question {block} {index}?"
                ),
                "work_id": work_id,
            }
            for index, work_id in enumerate(block_work_ids)
        ]
        items.reverse()
        public = source.self_hashed(
            {
                "block_id": block,
                "items": items,
                "schema": source.PUBLIC_BLOCK_SCHEMA,
                "study_id": ctl.STUDY_ID,
                "version": source.VERSION,
            }
        )
        public_values[block] = public
        public_raw[block] = _write_source_json(
            outputs.public_blocks()[block],
            public,
            mode=0o400 if block == "M_search" else 0o600,
        )
        if block in source.QREL_BLOCKS:
            rows: list[dict[str, object]] = []
            offset = 0
            for family in source.FAMILIES:
                family_count = source.DEFAULT_BLOCK_FAMILY_QUOTAS[block][
                    family
                ]
                for work_id in block_work_ids[offset : offset + family_count]:
                    first = global_ordinal % ctl.CORPUS_SIZE
                    second = (global_ordinal + 1) % ctl.CORPUS_SIZE
                    rows.append(
                        {
                            "family": family,
                            "gold_ordinals": sorted({first, second}),
                            "work_id": work_id,
                        }
                    )
                    global_ordinal += 1
                offset += family_count
            by_work = {row["work_id"]: row for row in rows}
            ordered = [by_work[item["work_id"]] for item in items]
            qrels = source.self_hashed(
                {
                    "block_id": block,
                    "qrels": ordered,
                    "schema": source.PRIVATE_QREL_SCHEMA,
                    "study_id": ctl.STUDY_ID,
                    "version": source.VERSION,
                }
            )
            qrel_values[block] = qrels
            qrel_raw[block] = _write_source_json(
                outputs.private_qrels()[block],
                qrels,
                mode=0o400,
            )

    public_bindings = {
        block: _binding(
            public_raw[block],
            public_values[block],
            mode=0o400 if block == "M_search" else 0o600,
            row_count=ctl.BLOCK_COUNTS[block],
        )
        for block in source.BLOCKS
    }
    qrel_bindings = {
        block: _binding(
            qrel_raw[block],
            qrel_values[block],
            mode=0o400,
            row_count=ctl.BLOCK_COUNTS[block],
        )
        for block in source.QREL_BLOCKS
    }
    corpus_binding = _binding(
        corpus_raw,
        corpus,
        mode=0o600,
        row_count=ctl.CORPUS_SIZE,
    )
    selected_qrel_count = 224
    receipt = source.self_hashed(
        {
            "artifact_binding": {
                "private_qrels": qrel_bindings,
                "private_selection_secret": {
                    "mode": "0600",
                    "selection_secret_commitment_sha256": hashlib.sha256(
                        secret
                    ).hexdigest(),
                    "selection_secret_persisted_publicly": False,
                    "size_bytes": len(secret),
                },
                "public_blocks": public_bindings,
                "public_corpus": corpus_binding,
            },
            "compiler_boundary": {
                "action_count": 0,
                "model_call_count": 0,
                "online_or_API_evaluation_count": 0,
                "score_count": 0,
            },
            "corpus_aggregate": {
                "arm_corpus_file_sha256": {
                    "Agent": corpus_binding["file_sha256"],
                    "RAW": corpus_binding["file_sha256"],
                    "official_HippoRAG": corpus_binding["file_sha256"],
                },
                "filler_unique_snippet_count": (
                    ctl.CORPUS_SIZE - selected_qrel_count
                ),
                "ordinal_text_row_count": ctl.CORPUS_SIZE,
                "rule": source.CORPUS_RULE,
                "selected_unique_qrel_count": selected_qrel_count,
            },
            "disjointness_aggregate": {
                "cross_block_component_overlap_count": 0,
                "cross_block_item_overlap_count": 0,
                "cross_block_normalized_query_overlap_count": 0,
                "maximum_selected_items_per_component": 1,
                "selected_component_count": 224,
                "selected_item_count": 224,
                "selected_normalized_query_count": 224,
            },
            "p0_binding": {
                "implementation": {
                    "sha256": source.P0_IMPLEMENTATION_SHA256,
                    "study_id": ctl.STUDY_ID,
                    "version": (
                        "bioasq_p0_public_source_qualification_v1"
                    ),
                },
                "private_manifest_file_sha256": (
                    source.P0_PRIVATE_MANIFEST_FILE_SHA256
                ),
                "private_manifest_self_sha256": (
                    source.P0_PRIVATE_MANIFEST_SELF_SHA256
                ),
                "public_audit_receipt_file_sha256": (
                    source.P0_PUBLIC_AUDIT_RECEIPT_FILE_SHA256
                ),
                "public_audit_receipt_self_sha256": (
                    source.P0_PUBLIC_AUDIT_RECEIPT_SELF_SHA256
                ),
                "safe_receipt_file_sha256": (
                    source.P0_SAFE_RECEIPT_FILE_SHA256
                ),
                "safe_receipt_self_sha256": (
                    source.P0_SAFE_RECEIPT_SELF_SHA256
                ),
            },
            "quota": {
                block: dict(source.DEFAULT_BLOCK_FAMILY_QUOTAS[block])
                for block in source.BLOCKS
            },
            "schema": source.SELECTION_RECEIPT_SCHEMA,
            "seal_contract": {
                "M_search_open_authorization": (
                    "controller_promotion_authorization_required"
                ),
                "M_search_presealed": True,
                "M_search_public_block_mode": "0400",
                "M_search_qrel_pack_mode": "0400",
                "other_late_qrel_pack_mode": "0400",
                "qrel_release_only_after_scored_block_actions_sealed": True,
            },
            "selection": {
                "block_order": list(source.BLOCKS),
                "family_order": list(source.FAMILIES),
                "rule": source.SELECTION_RULE,
                "selection_secret_commitment_sha256": hashlib.sha256(
                    secret
                ).hexdigest(),
                "selection_secret_file_create_count": 1,
                "selection_secret_generation_count": 1,
                "selection_secret_persisted_publicly": False,
                "work_id_rule": source.WORK_ID_RULE,
            },
            "source_access": {
                "file_sha256": source.OFFICIAL_SOURCE_SHA256,
                "formal_source_access_count": 1,
                "size_bytes": source.OFFICIAL_SOURCE_SIZE_BYTES,
                "source_hash_count": 1,
                "source_json_decode_count": 1,
                "source_open_count": 1,
            },
            "status": "selected_and_sealed",
            "study_id": ctl.STUDY_ID,
            "typed_core_binding": {
                "sha256": source.TYPED_CORE_SHA256,
                "study_id": ctl.STUDY_ID,
                "version": core.VERSION,
            },
            "version": source.VERSION,
        }
    )
    _write_source_json(
        outputs.safe_selection_receipt,
        receipt,
        mode=0o600,
    )
    return _Fixture(
        outputs=outputs,
        receipt=receipt,
        controller_root=controller_root,
        work_ids=work_ids,
    )


def _marker(fixture: _Fixture) -> Mapping[str, object]:
    marker = ctl.self_hashed(
        {
            "execution_binding_sha256": "a" * 64,
            ctl.NO_CHANGE_COUNT_KEY: 0,
            "schema": f"{ctl.VERSION}_one_shot_marker_v1",
            "study_id": ctl.STUDY_ID,
        }
    )
    _write_controller_json(
        fixture.controller_root / ctl.FORMAL_MARKER_FILENAME,
        marker,
    )
    return marker


def _claim(
    boundary: acquisition.SealedSourceAcquisitionBoundary,
    fixture: _Fixture,
) -> ctl.AcquisitionClaim:
    marker = _marker(fixture)
    return boundary.claim_formal_attempt(str(marker["self_sha256"]))


def _action_archive(
    fixture: _Fixture,
    block: ctl.BlockView,
    *,
    mode: int = 0o400,
) -> tuple[Path, Mapping[str, object]]:
    if block.block == "A_form":
        body: dict[str, object] = {
            "all_five_recipe_slates_sealed_before_qrels": True,
            "block": block.block,
            "block_view_sha256": block.view_sha256,
            "label_bearing_action_input_count": 0,
            "recipe_ids": list(core.RECIPE_IDS),
            "rows": [
                {"slate": {"synthetic": True}, "work_id": item.work_id}
                for item in block.items
            ],
            "schema": f"{ctl.VERSION}_A_form_private_action_archive_v1",
            "study_id": ctl.STUDY_ID,
        }
    else:
        body = {
            "E1_model_sha256": "b" * 64,
            "block": block.block,
            "block_view_sha256": block.view_sha256,
            "four_arms_sealed_before_qrels": True,
            "label_bearing_action_input_count": 0,
            "rows": [
                {
                    "arms": {"synthetic": True},
                    "work_id": item.work_id,
                }
                for item in block.items
            ],
            "schema": (
                f"{ctl.VERSION}_{block.block}_"
                "private_four_arm_action_archive_v1"
            ),
            "study_id": ctl.STUDY_ID,
        }
    archive = ctl.self_hashed(body)
    path = fixture.controller_root / f"{block.block}.actions.private.json"
    _write_controller_json(path, archive, mode=mode)
    return path, archive


def _authorization(
    fixture: _Fixture,
    claim: ctl.AcquisitionClaim,
) -> Mapping[str, object]:
    comparison = ctl.ExactPairedComparison(
        item_count=ctl.BLOCK_COUNTS["A_hold"],
        positive_count=ctl.BLOCK_COUNTS["A_hold"],
        negative_count=0,
        tie_count=0,
        net_utility=1,
        one_sided_exact_magnitude_preserving_tail=Fraction(1, 16),
    )
    authorization = ctl.self_hashed(
        {
            "A_hold_E1_minus_E0": comparison.payload(),
            "block_disjointness_commitment": (
                claim.block_disjointness_commitment
            ),
            "comparison_net_strictly_positive": True,
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth": (
                True
            ),
            "schema": (
                f"{ctl.VERSION}_"
                "M_search_materialization_authorization_v1"
            ),
            "status": "A_hold_E1_promoted",
            "study_id": ctl.STUDY_ID,
        }
    )
    _write_controller_json(
        fixture.controller_root / ctl.PROMOTION_AUTHORIZATION_FILENAME,
        authorization,
    )
    return authorization


def _instrument_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    opened: list[Path] = []
    original = acquisition._direct_file_bytes

    def counted(
        path: Path,
        *,
        mode: int,
        field: str,
    ) -> bytes:
        opened.append(path)
        return original(path, mode=mode, field=field)

    monkeypatch.setattr(acquisition, "_direct_file_bytes", counted)
    return opened


def test_one_read_public_qrel_and_conditional_m_lifecycle(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(posix_tmp)
    opened = _instrument_reads(monkeypatch)
    hippo = _HippoStub()
    boundary = acquisition.SealedSourceAcquisitionBoundary(
        outputs=fixture.outputs,
        selection_receipt=fixture.receipt,
        controller_root=fixture.controller_root,
        hippo_lane=hippo,
    )
    assert opened == [fixture.outputs.safe_selection_receipt]
    claim = _claim(boundary, fixture)
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="claimed twice",
    ):
        boundary.claim_formal_attempt(claim.claim_sha256)

    corpus = boundary.load_public_corpus(claim)
    assert len(corpus.passages) == ctl.CORPUS_SIZE
    assert hippo.build_call_count == 1
    assert hippo.corpus == corpus
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="corpus lifecycle",
    ):
        boundary.load_public_corpus(claim)
    assert opened.count(fixture.outputs.public_corpus) == 1

    blocks = {
        name: boundary.load_label_free_block(name)
        for name in ctl.INITIAL_BLOCKS
    }
    assert {
        name: len(block.items) for name, block in blocks.items()
    } == {
        "A_form": 96,
        "F_search": 32,
        "A_hold": 48,
    }
    assert hippo.build_call_count == 1
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="lifecycle",
    ):
        boundary.load_label_free_block("A_hold")
    assert opened.count(fixture.outputs.public_a_hold) == 1

    # F_search has no qrel channel.  No private qrel file is touched.
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="late-qrel lifecycle",
    ):
        boundary.release_qrels_after_action_seal(
            "F_search",
            fixture.controller_root / "F_search.actions.private.json",
            {},
        )
    assert sum(boundary.qrel_open_count.values()) == 0

    # A_form qrels remain unopened until a direct mode-0400 archive exists.
    absent_aform = fixture.controller_root / "A_form.actions.private.json"
    with pytest.raises(BioasqP1FormalRuntimeError):
        boundary.release_qrels_after_action_seal(
            "A_form",
            absent_aform,
            {},
        )
    assert opened.count(fixture.outputs.private_a_form_qrels) == 0
    aform_path, aform_archive = _action_archive(
        fixture,
        blocks["A_form"],
    )
    aform_pack = boundary.release_qrels_after_action_seal(
        "A_form",
        aform_path,
        aform_archive,
    )
    assert aform_pack.block == "A_form"
    assert aform_pack.action_archive_sha256 == aform_archive["self_sha256"]
    assert opened.count(fixture.outputs.private_a_form_qrels) == 1
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="late-qrel lifecycle",
    ):
        boundary.release_qrels_after_action_seal(
            "A_form",
            aform_path,
            aform_archive,
        )
    assert opened.count(fixture.outputs.private_a_form_qrels) == 1

    # Neither conditional public M nor its qrels can be opened early.
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="requires promotion authorization",
    ):
        boundary.load_label_free_block("M_search")
    assert opened.count(fixture.outputs.public_m_search) == 0
    assert opened.count(fixture.outputs.private_m_search_qrels) == 0

    authorization = _authorization(fixture, claim)
    m_block = boundary.load_label_free_block(
        "M_search",
        authorization,
    )
    assert len(m_block.items) == 48
    assert opened.count(fixture.outputs.public_m_search) == 1
    assert boundary.authorization_open_count == 1
    assert hippo.build_call_count == 1

    with pytest.raises(BioasqP1FormalRuntimeError):
        boundary.release_qrels_after_action_seal(
            "M_search",
            fixture.controller_root / "M_search.actions.private.json",
            {},
        )
    assert opened.count(fixture.outputs.private_m_search_qrels) == 0
    m_path, m_archive = _action_archive(fixture, m_block)
    m_pack = boundary.release_qrels_after_action_seal(
        "M_search",
        m_path,
        m_archive,
    )
    assert m_pack.block == "M_search"
    assert opened.count(fixture.outputs.private_m_search_qrels) == 1
    assert all(
        row.corpus_projection_sha256 == corpus.projection_sha256
        for row in m_pack.rows
    )

    ahold_path, ahold_archive = _action_archive(
        fixture,
        blocks["A_hold"],
    )
    ahold_pack = boundary.release_qrels_after_action_seal(
        "A_hold",
        ahold_path,
        ahold_archive,
    )
    assert ahold_pack.block == "A_hold"
    assert opened.count(fixture.outputs.private_a_hold_qrels) == 1
    assert boundary.public_open_count == {
        "corpus": 1,
        "A_form": 1,
        "F_search": 1,
        "A_hold": 1,
        "M_search": 1,
    }
    assert boundary.qrel_open_count == {
        "A_form": 1,
        "A_hold": 1,
        "M_search": 1,
    }


def test_invalid_authorization_never_opens_m_or_any_m_qrel(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(posix_tmp)
    opened = _instrument_reads(monkeypatch)
    boundary = acquisition.SealedSourceAcquisitionBoundary(
        outputs=fixture.outputs,
        selection_receipt=fixture.receipt,
        controller_root=fixture.controller_root,
        hippo_lane=_HippoStub(),
    )
    claim = _claim(boundary, fixture)
    boundary.load_public_corpus(claim)
    for block in ctl.INITIAL_BLOCKS:
        boundary.load_label_free_block(block)

    invalid = {
        "schema": (
            f"{ctl.VERSION}_M_search_materialization_authorization_v1"
        ),
        "status": "A_hold_E1_promoted",
    }
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="authorization schema",
    ):
        boundary.load_label_free_block("M_search", invalid)
    assert opened.count(fixture.outputs.public_m_search) == 0
    assert opened.count(fixture.outputs.private_m_search_qrels) == 0
    assert boundary.public_open_count["M_search"] == 0
    assert boundary.qrel_open_count["M_search"] == 0
    assert boundary.authorization_open_count == 0


def test_qrel_waits_for_exact_direct_mode_0400_action_archive(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(posix_tmp)
    opened = _instrument_reads(monkeypatch)
    boundary = acquisition.SealedSourceAcquisitionBoundary(
        outputs=fixture.outputs,
        selection_receipt=fixture.receipt,
        controller_root=fixture.controller_root,
        hippo_lane=_HippoStub(),
    )
    claim = _claim(boundary, fixture)
    boundary.load_public_corpus(claim)
    aform = boundary.load_label_free_block("A_form")
    action_path, action = _action_archive(
        fixture,
        aform,
        mode=0o600,
    )
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="mode-0400",
    ):
        boundary.release_qrels_after_action_seal(
            "A_form",
            action_path,
            action,
        )
    assert opened.count(fixture.outputs.private_a_form_qrels) == 0
    action_path.chmod(0o400)
    pack = boundary.release_qrels_after_action_seal(
        "A_form",
        action_path,
        action,
    )
    assert len(pack.rows) == ctl.BLOCK_COUNTS["A_form"]
    assert opened.count(fixture.outputs.private_a_form_qrels) == 1
    assert boundary.action_archive_open_count["A_form"] == 1


def test_selection_receipt_shared_corpus_binding_is_fail_closed(
    posix_tmp: Path,
) -> None:
    fixture = _fixture(posix_tmp)
    receipt = dict(fixture.receipt)
    receipt.pop("self_sha256")
    corpus = dict(receipt["corpus_aggregate"])  # type: ignore[arg-type]
    arms = dict(corpus["arm_corpus_file_sha256"])  # type: ignore[arg-type]
    arms["RAW"] = "f" * 64
    corpus["arm_corpus_file_sha256"] = arms
    receipt["corpus_aggregate"] = corpus
    receipt["self_sha256"] = stable_hash(receipt)
    raw = canonical_bytes(receipt, newline=True)
    _write_bytes(
        fixture.outputs.safe_selection_receipt,
        raw,
        mode=0o600,
    )
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="shared formal corpus",
    ):
        acquisition.SealedSourceAcquisitionBoundary(
            outputs=fixture.outputs,
            selection_receipt=receipt,
            controller_root=fixture.controller_root,
            hippo_lane=_HippoStub(),
        )
