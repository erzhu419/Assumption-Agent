"""One-shot, source-free formal controller for the frozen HiTab P1 study.

This module never imports a HiTab loader and never receives a source path,
split, table identifier, answer, family, or qrel before the corresponding
label-free action archive is durably sealed.  A trusted acquisition boundary
supplies opaque work IDs and source-free runtime views, then releases a
separate qrel pack only when given the already sealed action-archive hash.

The controller implements exactly one lifecycle:

* form and seal every A_form state/action registry, then release A_form qrels
  and fit E1 exactly once;
* form and seal all four A_hold arms, including a joined two-lane official
  HippoRAG queue, then release A_hold qrels;
* let only the preregistered E1-minus-E0 promotion authorize TEST's first
  decode and M_search materialization;
* if authorized, form and seal all four M_search arms before releasing its
  qrels and measuring the unchanged E1-minus-E0 L5 comparison.

All item-level material remains in mode-0400 private files.  The returned
terminal contains aggregate comparisons and opaque commitments only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import threading
from typing import Callable, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import hitab_p1_dmc1_core_v1 as core
from assumption_agent.benchmarks import hitab_p1_runtime_v1 as runtime


STUDY_ID = core.STUDY_ID
VERSION = "hitab_p1_formal_controller_v1"
FAMILIES = ("AGGREGATE", "COMPARATIVE", "SUPERLATIVE")
BLOCK_COUNTS = {"A_form": 108, "A_hold": 36, "M_search": 36}
FAMILY_COUNTS = {"A_form": 36, "A_hold": 12, "M_search": 12}
ALPHA = Fraction(1, 10)

FORMAL_MARKER_FILENAME = "formal.marker.json"
FORMAL_TERMINAL_FILENAME = "formal_terminal.json"
PROMOTION_AUTHORIZATION_FILENAME = "promotion.authorization.json"

_HEX64 = re.compile(r"[0-9a-f]{64}")
_WORK_ID = re.compile(r"hitab-work-v1-[0-9a-f]{64}")


class HitabP1FormalControllerError(RuntimeError):
    """The one-shot source-free formal lifecycle failed closed."""


class FormalAcquisitionBoundary(Protocol):
    """Trusted late-label boundary implemented outside this source-free file."""

    def claim_formal_attempt(
        self, formal_marker_sha256: str
    ) -> "AcquisitionClaim": ...

    def load_label_free_block(
        self,
        block: str,
        authorization: Mapping[str, object] | None = None,
    ) -> "BlockView": ...

    def release_qrels_after_action_seal(
        self,
        block: str,
        custody_path: Path,
        sealed_action_archive: Mapping[str, object],
    ) -> "QrelPack": ...


GPU0CacheReleaser = Callable[[], Mapping[str, object]]


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HitabP1FormalControllerError(
            "formal value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value:
        raise HitabP1FormalControllerError("self hash already exists")
    body = dict(value)
    body["self_sha256"] = stable_hash(body)
    return body


def _hex64(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise HitabP1FormalControllerError(f"{field} is not a SHA-256 digest")
    return value


def _work_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _WORK_ID.fullmatch(value) is None:
        raise HitabP1FormalControllerError(
            f"{field} is not a canonical opaque HiTab work ID"
        )
    return value


def _fraction_payload(value: Fraction) -> dict[str, int]:
    if not isinstance(value, Fraction):
        raise HitabP1FormalControllerError("comparison value is not exact")
    return {"denominator": value.denominator, "numerator": value.numerator}


def _comparison_payload(
    value: core.ExactPairedComparison,
) -> dict[str, object]:
    return {
        "negative_count": value.negative_count,
        "net_utility": _fraction_payload(value.net_utility),
        "one_sided_exact_magnitude_preserving_tail": _fraction_payload(
            value.reference_tail
        ),
        "positive_count": value.positive_count,
        "tie_count": value.tie_count,
    }


def _comparison_pass(value: core.ExactPairedComparison) -> bool:
    return value.net_utility > 0 and value.reference_tail <= ALPHA


def _validated_cache_release_receipt(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HitabP1FormalControllerError(
            "GPU0 cache release receipt is not an object"
        )
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        set(value)
        != {
            "model_offload_or_reload",
            "physical_gpu",
            "schema",
            "self_sha256",
            "study_id",
            "torch_cuda_empty_cache_called",
        }
        or value.get("schema")
        != "hitab_p1_gpu0_unused_cuda_cache_release_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("physical_gpu") != 0
        or value.get("torch_cuda_empty_cache_called") is not True
        or value.get("model_offload_or_reload") is not False
        or not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or not hmac.compare_digest(stable_hash(body), claimed)
    ):
        raise HitabP1FormalControllerError(
            "GPU0 cache release receipt drifted"
        )
    return dict(value)


@dataclass(frozen=True)
class AcquisitionClaim:
    """Only safe commitments released when the formal marker is claimed."""

    source_identity_commitment: str
    initial_selection_commitment: str
    claim_sha256: str

    def __post_init__(self) -> None:
        source = _hex64(
            self.source_identity_commitment,
            field="source identity commitment",
        )
        selection = _hex64(
            self.initial_selection_commitment,
            field="initial selection commitment",
        )
        expected = stable_hash(
            {
                "initial_selection_commitment": selection,
                "schema": f"{VERSION}_acquisition_claim_v1",
                "source_identity_commitment": source,
                "study_id": STUDY_ID,
            }
        )
        if not hmac.compare_digest(
            _hex64(self.claim_sha256, field="acquisition claim"), expected
        ):
            raise HitabP1FormalControllerError(
                "acquisition claim binding drifted"
            )

    @classmethod
    def create(
        cls,
        *,
        source_identity_commitment: str,
        initial_selection_commitment: str,
    ) -> "AcquisitionClaim":
        body = {
            "initial_selection_commitment": initial_selection_commitment,
            "schema": f"{VERSION}_acquisition_claim_v1",
            "source_identity_commitment": source_identity_commitment,
            "study_id": STUDY_ID,
        }
        return cls(
            source_identity_commitment=source_identity_commitment,
            initial_selection_commitment=initial_selection_commitment,
            claim_sha256=stable_hash(body),
        )


@dataclass(frozen=True)
class FormalItemView:
    """One opaque, label-free, source-free controller item."""

    work_id: str
    runtime_item: runtime.RuntimeItem

    def __post_init__(self) -> None:
        _work_id(self.work_id, field="work ID")
        if not isinstance(self.runtime_item, runtime.RuntimeItem):
            raise HitabP1FormalControllerError(
                "formal item runtime view drifted"
            )

    def private_payload(self) -> dict[str, object]:
        item = self.runtime_item
        return {
            "corpus_commitment": item.corpus_commitment,
            "ordered_unit_strings": list(item.ordered_unit_strings),
            "question": item.question,
            "typed_edges": [edge.payload() for edge in item.typed_edges],
            "unit_types": list(item.unit_types),
            "work_id": self.work_id,
        }


def _block_view_payload(
    block: str, items: Sequence[FormalItemView]
) -> dict[str, object]:
    return {
        "block": block,
        "items": [row.private_payload() for row in items],
    }


@dataclass(frozen=True)
class BlockView:
    """A complete block with no family or qrel field."""

    block: str
    items: tuple[FormalItemView, ...]
    view_sha256: str

    def __post_init__(self) -> None:
        if self.block not in BLOCK_COUNTS:
            raise HitabP1FormalControllerError("block view name drifted")
        if (
            not isinstance(self.items, tuple)
            or not self.items
            or any(not isinstance(row, FormalItemView) for row in self.items)
            or self.items
            != tuple(sorted(self.items, key=lambda row: row.work_id))
            or len({row.work_id for row in self.items}) != len(self.items)
        ):
            raise HitabP1FormalControllerError(
                "block view item order drifted"
            )
        expected = stable_hash(_block_view_payload(self.block, self.items))
        if not hmac.compare_digest(
            _hex64(self.view_sha256, field="block view"), expected
        ):
            raise HitabP1FormalControllerError("block view hash drifted")

    @classmethod
    def create(
        cls, block: str, items: Sequence[FormalItemView]
    ) -> "BlockView":
        checked = tuple(sorted(tuple(items), key=lambda row: row.work_id))
        return cls(
            block=block,
            items=checked,
            view_sha256=stable_hash(_block_view_payload(block, checked)),
        )


@dataclass(frozen=True)
class QrelRow:
    """One late qrel and its external result-grouping family."""

    work_id: str
    family: str
    proof: core.ProofDNF
    corpus_commitment: str
    qrel_ordinal_mapping_commitment: str

    def __post_init__(self) -> None:
        _work_id(self.work_id, field="qrel work ID")
        _hex64(self.corpus_commitment, field="qrel corpus commitment")
        if self.family not in FAMILIES:
            raise HitabP1FormalControllerError("qrel family drifted")
        if not isinstance(self.proof, core.ProofDNF):
            raise HitabP1FormalControllerError("qrel proof type drifted")
        if not hmac.compare_digest(
            self.proof.corpus_commitment, self.corpus_commitment
        ):
            raise HitabP1FormalControllerError(
                "qrel proof corpus commitment drifted"
            )
        # The official HiTab [ANSWER] field is a single coordinate proof.
        # Each annotated coordinate is a separate singleton requirement.
        if (
            len(self.proof.alternatives) != 1
            or not 1 <= len(self.proof.alternatives[0]) <= core.TOP_K
            or any(len(bucket) != 1 for bucket in self.proof.alternatives[0])
        ):
            raise HitabP1FormalControllerError(
                "formal qrel must be one alternative of singleton buckets"
            )
        ordinals = tuple(
            bucket[0] for bucket in self.proof.alternatives[0]
        )
        if len(set(ordinals)) != len(ordinals):
            raise HitabP1FormalControllerError(
                "formal qrel repeats an annotated coordinate"
            )
        expected = self.proof.ordinal_mapping_commitment
        if not hmac.compare_digest(
            _hex64(
                self.qrel_ordinal_mapping_commitment,
                field="qrel ordinal mapping commitment",
            ),
            expected,
        ):
            raise HitabP1FormalControllerError(
                "qrel ordinal mapping commitment drifted"
            )


def _qrel_pack_payload(
    block: str,
    action_archive_sha256: str,
    rows: Sequence[QrelRow],
) -> dict[str, object]:
    return {
        "action_archive_sha256": action_archive_sha256,
        "block": block,
        "rows": [
            {
                "corpus_commitment": row.corpus_commitment,
                "family": row.family,
                "proof": row.proof.payload(),
                "qrel_ordinal_mapping_commitment": (
                    row.qrel_ordinal_mapping_commitment
                ),
                "work_id": row.work_id,
            }
            for row in rows
        ],
    }


@dataclass(frozen=True)
class QrelPack:
    """A late pack cryptographically bound to the prior action archive."""

    block: str
    action_archive_sha256: str
    rows: tuple[QrelRow, ...]
    pack_sha256: str

    def __post_init__(self) -> None:
        if self.block not in BLOCK_COUNTS:
            raise HitabP1FormalControllerError("qrel block name drifted")
        archive = _hex64(
            self.action_archive_sha256, field="qrel action archive"
        )
        if (
            not isinstance(self.rows, tuple)
            or not self.rows
            or any(not isinstance(row, QrelRow) for row in self.rows)
            or self.rows
            != tuple(sorted(self.rows, key=lambda row: row.work_id))
            or len({row.work_id for row in self.rows}) != len(self.rows)
        ):
            raise HitabP1FormalControllerError("qrel pack row order drifted")
        expected = stable_hash(
            _qrel_pack_payload(self.block, archive, self.rows)
        )
        if not hmac.compare_digest(
            _hex64(self.pack_sha256, field="qrel pack"), expected
        ):
            raise HitabP1FormalControllerError("qrel pack hash drifted")

    @classmethod
    def create(
        cls,
        *,
        block: str,
        action_archive_sha256: str,
        rows: Sequence[QrelRow],
    ) -> "QrelPack":
        checked = tuple(sorted(tuple(rows), key=lambda row: row.work_id))
        return cls(
            block=block,
            action_archive_sha256=action_archive_sha256,
            rows=checked,
            pack_sha256=stable_hash(
                _qrel_pack_payload(
                    block, action_archive_sha256, checked
                )
            ),
        )


@dataclass(frozen=True)
class _SealedFile:
    path: Path
    self_sha256: str
    file_sha256: str
    value: Mapping[str, object]


@dataclass(frozen=True)
class _CompiledItem:
    item: FormalItemView
    compiled: runtime.CompiledRuntime
    registry: core.SealedAFormRegistry | None = None
    raw: tuple[int, ...] | None = None
    e0: tuple[int, ...] | None = None
    e1: tuple[int, ...] | None = None
    hippo: runtime.OfficialHippoAction | None = None


@dataclass(frozen=True)
class _FourArmFormation:
    rows: tuple[_CompiledItem, ...]
    gpu0_cache_release_receipt: Mapping[str, object]


@dataclass(frozen=True)
class _ScoredBlock:
    comparison_e1_e0: core.ExactPairedComparison
    comparison_e1_raw: core.ExactPairedComparison
    comparison_e1_hippo: core.ExactPairedComparison
    family_e1_raw: Mapping[str, core.ExactPairedComparison]
    family_e1_hippo: Mapping[str, core.ExactPairedComparison]
    arm_total_utility: Mapping[str, Fraction]
    arm_complete_proof_count: Mapping[str, int]
    action_set_difference_count: Mapping[str, int]
    score_archive: _SealedFile


def _exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise HitabP1FormalControllerError(
            f"exclusive formal file already exists or is unsafe: {path.name}"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short formal write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _seal_json(
    root: Path,
    filename: str,
    body: Mapping[str, object],
) -> _SealedFile:
    value = self_hashed(body)
    raw = canonical_bytes(value)
    path = root / filename
    _exclusive_bytes(path, raw)
    return _SealedFile(
        path=path,
        self_sha256=str(value["self_sha256"]),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        value=value,
    )


def _validate_block(block: BlockView, *, expected: str) -> None:
    if (
        not isinstance(block, BlockView)
        or block.block != expected
        or len(block.items) != BLOCK_COUNTS[expected]
    ):
        raise HitabP1FormalControllerError(
            f"{expected} label-free block count drifted"
        )


def _validate_qrels(
    pack: QrelPack,
    *,
    block: BlockView,
    action_archive_sha256: str,
) -> Mapping[str, QrelRow]:
    if (
        not isinstance(pack, QrelPack)
        or pack.block != block.block
        or not hmac.compare_digest(
            pack.action_archive_sha256, action_archive_sha256
        )
        or len(pack.rows) != len(block.items)
    ):
        raise HitabP1FormalControllerError("late qrel pack binding drifted")
    by_work = {row.work_id: row for row in pack.rows}
    if set(by_work) != {row.work_id for row in block.items}:
        raise HitabP1FormalControllerError(
            "late qrel pack work coverage drifted"
        )
    family_counts = {family: 0 for family in FAMILIES}
    for view_row in block.items:
        qrel = by_work[view_row.work_id]
        family_counts[qrel.family] += 1
        if not hmac.compare_digest(
            qrel.corpus_commitment,
            view_row.runtime_item.corpus_commitment,
        ):
            raise HitabP1FormalControllerError(
                "late qrel corpus binding drifted"
            )
        for bucket in qrel.proof.alternatives[0]:
            ordinal = bucket[0]
            if not 0 <= ordinal < len(
                view_row.runtime_item.ordered_unit_strings
            ):
                raise HitabP1FormalControllerError(
                    "late qrel ordinal escaped the sealed corpus"
                )
    expected_per_family = FAMILY_COUNTS[block.block]
    if family_counts != {
        family: expected_per_family for family in FAMILIES
    }:
        raise HitabP1FormalControllerError(
            "late qrel family quota drifted"
        )
    return by_work


def _compile_block(
    block: BlockView,
    *,
    planner_runner: runtime.PlannerByteRunner,
    cross_encoder_scorer: runtime.CrossEncoderPairScorer,
    minilm_encoder: runtime.MiniLMEncoder,
) -> tuple[_CompiledItem, ...]:
    rows: list[_CompiledItem] = []
    for row in block.items:
        compiled = runtime.compile_runtime(
            row.runtime_item,
            planner_runner=planner_runner,
            cross_encoder_scorer=cross_encoder_scorer,
            minilm_encoder=minilm_encoder,
            physical_gpu=runtime.AGENT_FORMATION_PHYSICAL_GPU,
        )
        if not hmac.compare_digest(
            compiled.view.corpus_commitment,
            row.runtime_item.corpus_commitment,
        ):
            raise HitabP1FormalControllerError(
                "compiled corpus commitment drifted"
            )
        rows.append(_CompiledItem(item=row, compiled=compiled))
    return tuple(rows)


def _aform_action_body(
    block: BlockView, rows: Sequence[_CompiledItem]
) -> dict[str, object]:
    return {
        "block": "A_form",
        "block_view_sha256": block.view_sha256,
        "item_count": len(rows),
        "registry_stage_complete": True,
        "records": [
            {
                "corpus_commitment": row.item.runtime_item.corpus_commitment,
                "registry": core.registry_payload(row.registry),
                "tensor_sha256": row.compiled.tensor_sha256,
                "work_id": row.item.work_id,
            }
            for row in rows
            if row.registry is not None
        ],
        "schema": f"{VERSION}_A_form_label_free_action_archive_v1",
        "study_id": STUDY_ID,
    }


def _form_aform(
    block: BlockView,
    *,
    execution_binding_sha256: str,
    planner_runner: runtime.PlannerByteRunner,
    cross_encoder_scorer: runtime.CrossEncoderPairScorer,
    minilm_encoder: runtime.MiniLMEncoder,
) -> tuple[_CompiledItem, ...]:
    compiled = _compile_block(
        block,
        planner_runner=planner_runner,
        cross_encoder_scorer=cross_encoder_scorer,
        minilm_encoder=minilm_encoder,
    )
    exploration_key = hashlib.sha256(
        canonical_bytes(
            {
                "execution_binding_sha256": execution_binding_sha256,
                "schema": f"{VERSION}_A_form_exploration_key_v1",
                "study_id": STUDY_ID,
            }
        )
    ).digest()
    return tuple(
        _CompiledItem(
            item=row.item,
            compiled=row.compiled,
            registry=core.build_and_seal_aform_registry(
                row.compiled.view, exploration_key=exploration_key
            ),
        )
        for row in compiled
    )


def _proof_private_payload(row: QrelRow) -> dict[str, object]:
    return {
        "corpus_commitment": row.corpus_commitment,
        "family": row.family,
        "proof": row.proof.payload(),
        "qrel_ordinal_mapping_commitment": (
            row.qrel_ordinal_mapping_commitment
        ),
        "work_id": row.work_id,
    }


def _seal_qrel_pack(
    root: Path,
    pack: QrelPack,
) -> _SealedFile:
    return _seal_json(
        root,
        f"{pack.block}.qrels.private.json",
        {
            "action_archive_sha256": pack.action_archive_sha256,
            "block": pack.block,
            "pack_sha256": pack.pack_sha256,
            "rows": [_proof_private_payload(row) for row in pack.rows],
            "schema": f"{VERSION}_{pack.block}_late_qrel_archive_v1",
            "study_id": STUDY_ID,
        },
    )


def _fit_e1_once(
    rows: Sequence[_CompiledItem],
    qrels: Mapping[str, QrelRow],
) -> core.E1Model:
    labelled = []
    for row in rows:
        if row.registry is None:
            raise HitabP1FormalControllerError(
                "A_form registry is incomplete"
            )
        labelled.append(
            core.label_sealed_registry(
                row.registry, qrels[row.item.work_id].proof
            )
        )
    return core.fit_e1(tuple(labelled))


def _formation_bindings(
    rows: Sequence[_CompiledItem],
    qrels: Mapping[str, QrelRow],
) -> tuple[dict[str, str], ...]:
    bindings: list[dict[str, str]] = []
    for row in rows:
        if row.registry is None:
            raise HitabP1FormalControllerError(
                "A_form registry is incomplete at model seal"
            )
        qrel = qrels[row.item.work_id]
        if not (
            row.registry.corpus_commitment
            == qrel.corpus_commitment
            == qrel.proof.corpus_commitment
        ):
            raise HitabP1FormalControllerError(
                "A_form work/corpus/qrel lineage drifted"
            )
        bindings.append(
            {
                "corpus_commitment": qrel.corpus_commitment,
                "qrel_ordinal_mapping_commitment": (
                    qrel.qrel_ordinal_mapping_commitment
                ),
                "registry_seal_sha256": row.registry.seal_sha256,
                "work_id": row.item.work_id,
            }
        )
    result = tuple(sorted(bindings, key=lambda value: value["work_id"]))
    if len(result) != BLOCK_COUNTS["A_form"]:
        raise HitabP1FormalControllerError(
            "A_form model lineage count drifted"
        )
    return result


def _validate_model_formation_bindings(
    model: core.E1Model,
    bindings: Sequence[Mapping[str, str]],
) -> None:
    corpus = [row["corpus_commitment"] for row in bindings]
    qrel = [
        row["qrel_ordinal_mapping_commitment"] for row in bindings
    ]
    pairs = [
        [
            row["corpus_commitment"],
            row["qrel_ordinal_mapping_commitment"],
        ]
        for row in bindings
    ]
    if (
        model.training_registry_count != len(bindings)
        or model.training_corpus_set_commitment
        != core.stable_hash(sorted(corpus))
        or model.training_qrel_mapping_set_commitment
        != core.stable_hash(sorted(set(qrel)))
        or model.training_corpus_qrel_binding_set_commitment
        != core.stable_hash(sorted(pairs))
    ):
        raise HitabP1FormalControllerError(
            "E1 model formation lineage commitment drifted"
        )


def _four_arm_action_body(
    block: BlockView,
    rows: Sequence[_CompiledItem],
    *,
    e1_model_sha256: str,
    gpu0_cache_release_receipt: Mapping[str, object],
) -> dict[str, object]:
    checked_model_sha256 = _hex64(
        e1_model_sha256, field="frozen E1 model archive"
    )
    records: list[dict[str, object]] = []
    for row in rows:
        if (
            row.raw is None
            or row.e0 is None
            or row.e1 is None
            or row.hippo is None
        ):
            raise HitabP1FormalControllerError(
                f"{block.block} four-arm row is incomplete"
            )
        commitment = row.item.runtime_item.corpus_commitment
        if not (
            commitment
            == row.compiled.view.corpus_commitment
            == row.hippo.corpus_commitment
        ):
            raise HitabP1FormalControllerError(
                f"{block.block} four-arm corpus binding drifted"
            )
        records.append(
            {
                "arms": {
                    "E0": {
                        "corpus_commitment": commitment,
                        "top5_ordinals": list(row.e0),
                    },
                    "E1": {
                        "corpus_commitment": commitment,
                        "top5_ordinals": list(row.e1),
                    },
                    "HippoRAG": {
                        "complete_rank_sha256": (
                            row.hippo.complete_rank_sha256
                        ),
                        "corpus_commitment": commitment,
                        "input_sha256": row.hippo.input_sha256,
                        "output_sha256": row.hippo.output_sha256,
                        "physical_gpu": row.hippo.physical_gpu,
                        "top5_ordinals": list(row.hippo.top5_ordinals),
                    },
                    "RAW": {
                        "corpus_commitment": commitment,
                        "top5_ordinals": list(row.raw),
                    },
                },
                "tensor_sha256": row.compiled.tensor_sha256,
                "work_id": row.item.work_id,
            }
        )
    return {
        "block": block.block,
        "block_view_sha256": block.view_sha256,
        "four_arm_corpus_commitment_exact": True,
        "gpu0_unused_cuda_cache_release_receipt": (
            _validated_cache_release_receipt(
                gpu0_cache_release_receipt
            )
        ),
        "hipporag_queue_joined_before_archive": True,
        "item_count": len(records),
        "e1_model_sha256": checked_model_sha256,
        "records": records,
        "schema": f"{VERSION}_{block.block}_four_arm_action_archive_v1",
        "study_id": STUDY_ID,
    }


def _form_four_arms(
    block: BlockView,
    model: core.E1Model,
    *,
    planner_runner: runtime.PlannerByteRunner,
    cross_encoder_scorer: runtime.CrossEncoderPairScorer,
    minilm_encoder: runtime.MiniLMEncoder,
    hippo_runner: runtime.OfficialHippoByteRunner,
    gpu0_cache_releaser: GPU0CacheReleaser,
) -> _FourArmFormation:
    if block.block not in {"A_hold", "M_search"}:
        raise HitabP1FormalControllerError(
            "four-arm formation block drifted"
        )
    hippo_output: list[runtime.OfficialHippoAction | None] = [
        None
    ] * len(block.items)
    release_gpu0 = threading.Event()
    gpu1_ack_or_terminal = threading.Event()
    gpu1_launch_acknowledged = threading.Event()
    abort = threading.Event()

    def hippo_lane(physical_gpu: int) -> None:
        try:
            if physical_gpu == 0:
                release_gpu0.wait()
                if abort.is_set():
                    return
            for index in range(physical_gpu, len(block.items), 2):
                if abort.is_set():
                    return
                item = block.items[index].runtime_item

                def acknowledge_launch() -> None:
                    if physical_gpu == 1 and not (
                        gpu1_launch_acknowledged.is_set()
                    ):
                        gpu1_launch_acknowledged.set()
                        gpu1_ack_or_terminal.set()

                hippo_output[index] = runtime.run_official_hippo(
                    item.question,
                    item.ordered_unit_strings,
                    hippo_runner,
                    physical_gpu=physical_gpu,
                    launch_ack=acknowledge_launch,
                )
        finally:
            if physical_gpu == 1:
                # Wake the coordinator on a pre-ack terminal failure; it must
                # never wait forever or treat queue liveness as process launch.
                gpu1_ack_or_terminal.set()

    compiled_with_actions: tuple[_CompiledItem, ...] | None = None
    cache_release_receipt: Mapping[str, object] | None = None
    formation_error: Exception | None = None
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix=(
            f"hitab-{block.block}-official-hippo-physical-gpu"
        ),
    ) as executor:
        futures = (
            executor.submit(hippo_lane, 0),
            executor.submit(hippo_lane, 1),
        )
        gpu1_ack_or_terminal.wait()
        if not gpu1_launch_acknowledged.is_set():
            formation_error = HitabP1FormalControllerError(
                "GPU1 HippoRAG terminated before launch acknowledgement"
            )
            abort.set()
            release_gpu0.set()
        else:
            try:
                compiled = _compile_block(
                    block,
                    planner_runner=planner_runner,
                    cross_encoder_scorer=cross_encoder_scorer,
                    minilm_encoder=minilm_encoder,
                )
                compiled_with_actions = tuple(
                    _CompiledItem(
                        item=row.item,
                        compiled=row.compiled,
                        raw=row.compiled.raw_top5,
                        e0=core.select_e0(row.compiled.view),
                        e1=core.select_e1(row.compiled.view, model),
                    )
                    for row in compiled
                )
                cache_release_receipt = _validated_cache_release_receipt(
                    gpu0_cache_releaser()
                )
            except Exception as exc:
                formation_error = exc
                abort.set()
            finally:
                # Physical GPU0 remains unavailable to HippoRAG until every
                # planner/CE/MiniLM tensor and RAW/E0/E1 action is complete.
                release_gpu0.set()
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                if formation_error is None:
                    formation_error = exc
                abort.set()

    if formation_error is not None:
        raise formation_error
    if (
        compiled_with_actions is None
        or cache_release_receipt is None
        or any(row is None for row in hippo_output)
    ):
        raise HitabP1FormalControllerError(
            "joined official HippoRAG output is incomplete"
        )
    return _FourArmFormation(
        rows=tuple(
            _CompiledItem(
                item=row.item,
                compiled=row.compiled,
                raw=row.raw,
                e0=row.e0,
                e1=row.e1,
                hippo=hippo_output[index],
            )
            for index, row in enumerate(compiled_with_actions)
        ),
        gpu0_cache_release_receipt=cache_release_receipt,
    )


def _utility(
    action: Sequence[int],
    row: _CompiledItem,
    qrel: QrelRow,
) -> Fraction:
    return core.set_utility(
        tuple(sorted(action)),
        qrel.proof,
        unit_count=row.compiled.view.unit_count,
    )


def _score_four_arm_block(
    root: Path,
    block: BlockView,
    rows: Sequence[_CompiledItem],
    qrels: Mapping[str, QrelRow],
) -> _ScoredBlock:
    private_rows: list[dict[str, object]] = []
    utility_e0: list[Fraction] = []
    utility_e1: list[Fraction] = []
    utility_raw: list[Fraction] = []
    utility_hippo: list[Fraction] = []
    family_indices = {family: [] for family in FAMILIES}
    action_set_difference_count = {
        "E1_vs_E0": 0,
        "E1_vs_HippoRAG": 0,
        "E1_vs_RAW": 0,
    }
    for index, row in enumerate(rows):
        qrel = qrels[row.item.work_id]
        if (
            row.raw is None
            or row.e0 is None
            or row.e1 is None
            or row.hippo is None
        ):
            raise HitabP1FormalControllerError(
                f"{block.block} action row is incomplete at score time"
            )
        values = {
            "E0": _utility(row.e0, row, qrel),
            "E1": _utility(row.e1, row, qrel),
            "HippoRAG": _utility(
                row.hippo.top5_ordinals, row, qrel
            ),
            "RAW": _utility(row.raw, row, qrel),
        }
        utility_e0.append(values["E0"])
        utility_e1.append(values["E1"])
        utility_raw.append(values["RAW"])
        utility_hippo.append(values["HippoRAG"])
        e1_set = frozenset(row.e1)
        action_set_difference_count["E1_vs_E0"] += int(
            e1_set != frozenset(row.e0)
        )
        action_set_difference_count["E1_vs_HippoRAG"] += int(
            e1_set != frozenset(row.hippo.top5_ordinals)
        )
        action_set_difference_count["E1_vs_RAW"] += int(
            e1_set != frozenset(row.raw)
        )
        family_indices[qrel.family].append(index)
        private_rows.append(
            {
                "family": qrel.family,
                "utilities": {
                    key: _fraction_payload(value)
                    for key, value in values.items()
                },
                "work_id": row.item.work_id,
            }
        )

    e1_e0 = core.compare_paired(utility_e1, utility_e0)
    e1_raw = core.compare_paired(utility_e1, utility_raw)
    e1_hippo = core.compare_paired(utility_e1, utility_hippo)
    family_raw = {
        family: core.compare_paired(
            [utility_e1[index] for index in indices],
            [utility_raw[index] for index in indices],
        )
        for family, indices in family_indices.items()
    }
    family_hippo = {
        family: core.compare_paired(
            [utility_e1[index] for index in indices],
            [utility_hippo[index] for index in indices],
        )
        for family, indices in family_indices.items()
    }
    utility_by_arm = {
        "E0": utility_e0,
        "E1": utility_e1,
        "HippoRAG": utility_hippo,
        "RAW": utility_raw,
    }
    arm_total_utility = {
        arm: sum(values, Fraction(0, 1))
        for arm, values in utility_by_arm.items()
    }
    arm_complete_proof_count = {
        arm: sum(value == 2 for value in values)
        for arm, values in utility_by_arm.items()
    }
    archive = _seal_json(
        root,
        f"{block.block}.scores.private.json",
        {
            "aggregate": {
                "E1_minus_E0": _comparison_payload(e1_e0),
                "E1_minus_HippoRAG": _comparison_payload(e1_hippo),
                "E1_minus_RAW": _comparison_payload(e1_raw),
            },
            "block": block.block,
            "descriptive": {
                "action_set_difference_count": (
                    action_set_difference_count
                ),
                "arm_complete_proof_count": arm_complete_proof_count,
                "arm_total_exact_utility": {
                    arm: _fraction_payload(value)
                    for arm, value in arm_total_utility.items()
                },
                "outside_RAW_count": action_set_difference_count[
                    "E1_vs_RAW"
                ],
            },
            "family": {
                family: {
                    "E1_minus_HippoRAG": _comparison_payload(
                        family_hippo[family]
                    ),
                    "E1_minus_RAW": _comparison_payload(
                        family_raw[family]
                    ),
                }
                for family in FAMILIES
            },
            "rows": private_rows,
            "schema": f"{VERSION}_{block.block}_private_score_archive_v1",
            "study_id": STUDY_ID,
        },
    )
    return _ScoredBlock(
        comparison_e1_e0=e1_e0,
        comparison_e1_raw=e1_raw,
        comparison_e1_hippo=e1_hippo,
        family_e1_raw=family_raw,
        family_e1_hippo=family_hippo,
        arm_total_utility=arm_total_utility,
        arm_complete_proof_count=arm_complete_proof_count,
        action_set_difference_count=action_set_difference_count,
        score_archive=archive,
    )


def _promotion_authorization(
    claim: AcquisitionClaim,
    promotion: core.ExactPairedComparison,
) -> dict[str, object]:
    return self_hashed(
        {
            "aggregate_exact_utility_net_strictly_positive": (
                promotion.net_utility > 0
            ),
            "comparison": "E1_minus_E0",
            "initial_selection_commitment": (
                claim.initial_selection_commitment
            ),
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth": (
                promotion.reference_tail <= ALPHA
            ),
            "schema": "hitab_p1_test_first_decode_authorization_v1",
            "source_identity_commitment": (
                claim.source_identity_commitment
            ),
            "status": "A_hold_E1_promoted",
            "study_id": STUDY_ID,
        }
    )


def _safe_block_result(value: _ScoredBlock) -> dict[str, object]:
    reality_passed = (
        _comparison_pass(value.comparison_e1_raw)
        and _comparison_pass(value.comparison_e1_hippo)
        and all(
            value.family_e1_raw[family].net_utility > 0
            and value.family_e1_hippo[family].net_utility > 0
            for family in FAMILIES
        )
    )
    return {
        "aggregate": {
            "E1_minus_E0": _comparison_payload(
                value.comparison_e1_e0
            ),
            "E1_minus_HippoRAG": _comparison_payload(
                value.comparison_e1_hippo
            ),
            "E1_minus_RAW": _comparison_payload(
                value.comparison_e1_raw
            ),
        },
        "descriptive": {
            "action_set_difference_count": dict(
                value.action_set_difference_count
            ),
            "arm_complete_proof_count": dict(
                value.arm_complete_proof_count
            ),
            "arm_total_exact_utility": {
                arm: _fraction_payload(total)
                for arm, total in value.arm_total_utility.items()
            },
            "outside_RAW_count": value.action_set_difference_count[
                "E1_vs_RAW"
            ],
        },
        "family": {
            family: {
                "E1_minus_HippoRAG": _comparison_payload(
                    value.family_e1_hippo[family]
                ),
                "E1_minus_RAW": _comparison_payload(
                    value.family_e1_raw[family]
                ),
            }
            for family in FAMILIES
        },
        "promotion_passed": _comparison_pass(
            value.comparison_e1_e0
        ),
        "reality_primary_passed": reality_passed,
    }


def _terminal_body(
    *,
    execution_binding_sha256: str,
    marker_sha256: str,
    claim: AcquisitionClaim,
    aform_action: _SealedFile,
    aform_qrel: _SealedFile,
    model_archive: _SealedFile,
    ahold_action: _SealedFile,
    ahold_qrel: _SealedFile,
    ahold_score: _ScoredBlock,
    m_action: _SealedFile | None,
    m_qrel: _SealedFile | None,
    m_score: _ScoredBlock | None,
) -> dict[str, object]:
    promotion = _comparison_pass(ahold_score.comparison_e1_e0)
    archives: dict[str, object] = {
        "A_form_action_archive_sha256": aform_action.self_sha256,
        "A_form_qrel_archive_sha256": aform_qrel.self_sha256,
        "A_hold_action_archive_sha256": ahold_action.self_sha256,
        "A_hold_qrel_archive_sha256": ahold_qrel.self_sha256,
        "A_hold_score_archive_sha256": (
            ahold_score.score_archive.self_sha256
        ),
        "E1_model_archive_sha256": model_archive.self_sha256,
    }
    if m_action is not None and m_qrel is not None and m_score is not None:
        archives.update(
            {
                "M_search_action_archive_sha256": m_action.self_sha256,
                "M_search_qrel_archive_sha256": m_qrel.self_sha256,
                "M_search_score_archive_sha256": (
                    m_score.score_archive.self_sha256
                ),
            }
        )
    return {
        "A_hold": _safe_block_result(ahold_score),
        "M_search": (
            {
                "descriptive": _safe_block_result(m_score)[
                    "descriptive"
                ],
                "L5_E1_minus_E0": _comparison_payload(
                    m_score.comparison_e1_e0
                ),
                "L5_passed": _comparison_pass(
                    m_score.comparison_e1_e0
                ),
                "opened_after_promotion": True,
            }
            if m_score is not None
            else {
                "descriptive": None,
                "L5_E1_minus_E0": None,
                "L5_passed": None,
                "opened_after_promotion": False,
            }
        ),
        "acquisition_claim_sha256": claim.claim_sha256,
        "aggregate_only_public_terminal": True,
        "archive_commitments": archives,
        "execution_binding_sha256": execution_binding_sha256,
        "formal_marker_sha256": marker_sha256,
        "item_query_unit_qrel_or_per_item_score_values_published": False,
        "online_or_API_evaluator_calls": 0,
        "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
        "schema": f"{VERSION}_safe_terminal_v1",
        "status": (
            "terminal_complete_after_A_hold_promotion_and_M_search"
            if promotion
            else "terminal_A_hold_E1_not_promoted_M_search_unopened"
        ),
        "study_id": STUDY_ID,
    }


def _write_failure_terminal(
    root: Path,
    *,
    execution_binding_sha256: str,
    marker_sha256: str,
    stage: str,
    exc: Exception,
) -> None:
    body = {
        "aggregate_only_public_terminal": True,
        "execution_binding_sha256": execution_binding_sha256,
        "failure_exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8", errors="replace")
        ).hexdigest(),
        "failure_exception_type_sha256": hashlib.sha256(
            type(exc).__name__.encode("ascii", errors="replace")
        ).hexdigest(),
        "failure_stage": stage,
        "formal_marker_sha256": marker_sha256,
        "item_query_unit_qrel_or_per_item_score_values_published": False,
        "online_or_API_evaluator_calls": 0,
        "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
        "schema": f"{VERSION}_safe_failure_terminal_v1",
        "status": "terminal_formal_failure_no_retry",
        "study_id": STUDY_ID,
    }
    try:
        _seal_json(root, FORMAL_TERMINAL_FILENAME, body)
    except Exception:
        # Preserve the original formal error.  A pre-existing terminal or a
        # filesystem failure must never turn into a replay or overwrite.
        pass


def run_formal_controller(
    *,
    work_root: Path,
    execution_binding_sha256: str,
    acquisition: FormalAcquisitionBoundary,
    planner_runner: runtime.PlannerByteRunner,
    cross_encoder_scorer: runtime.CrossEncoderPairScorer,
    minilm_encoder: runtime.MiniLMEncoder,
    hippo_runner: runtime.OfficialHippoByteRunner,
    gpu0_cache_releaser: GPU0CacheReleaser,
) -> Mapping[str, object]:
    """Execute the single frozen formal lifecycle and return its safe terminal."""

    binding = _hex64(
        execution_binding_sha256, field="execution binding"
    )
    root = Path(work_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise HitabP1FormalControllerError("formal work root is unsafe")
    os.chmod(root, 0o700)
    marker = self_hashed(
        {
            "execution_binding_sha256": binding,
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": f"{VERSION}_one_shot_marker_v1",
            "study_id": STUDY_ID,
        }
    )
    marker_raw = canonical_bytes(marker)
    _exclusive_bytes(root / FORMAL_MARKER_FILENAME, marker_raw)
    marker_sha256 = str(marker["self_sha256"])
    stage = "claim_acquisition"
    try:
        claim = acquisition.claim_formal_attempt(marker_sha256)
        if not isinstance(claim, AcquisitionClaim):
            raise HitabP1FormalControllerError(
                "acquisition boundary claim type drifted"
            )

        stage = "load_A_form_label_free"
        aform = acquisition.load_label_free_block("A_form", None)
        _validate_block(aform, expected="A_form")
        stage = "form_and_seal_A_form_actions"
        aform_rows = _form_aform(
            aform,
            execution_binding_sha256=binding,
            planner_runner=planner_runner,
            cross_encoder_scorer=cross_encoder_scorer,
            minilm_encoder=minilm_encoder,
        )
        aform_action = _seal_json(
            root,
            "A_form.actions.private.json",
            _aform_action_body(aform, aform_rows),
        )
        if any(
            token in aform_action.path.read_bytes().lower()
            for token in (b'"qrel"', b'"gold"', b'"proof"', b'"family"')
        ):
            raise HitabP1FormalControllerError(
                "A_form prelabel archive leaked a label channel"
            )

        stage = "release_A_form_qrels_after_action_seal"
        aform_pack = acquisition.release_qrels_after_action_seal(
            "A_form", aform_action.path, aform_action.value
        )
        aform_qrels = _validate_qrels(
            aform_pack,
            block=aform,
            action_archive_sha256=aform_action.self_sha256,
        )
        aform_qrel_archive = _seal_qrel_pack(root, aform_pack)
        stage = "fit_and_seal_E1_once"
        model = _fit_e1_once(aform_rows, aform_qrels)
        formation_bindings = _formation_bindings(
            aform_rows, aform_qrels
        )
        _validate_model_formation_bindings(model, formation_bindings)
        model_archive = _seal_json(
            root,
            "E1.model.private.json",
            {
                "formation_binding_set_commitment": stable_hash(
                    formation_bindings
                ),
                "formation_bindings": list(formation_bindings),
                "formation_item_count": len(formation_bindings),
                "model": core.model_payload(model),
                "schema": f"{VERSION}_frozen_E1_model_v1",
                "study_id": STUDY_ID,
            },
        )

        stage = "load_A_hold_label_free"
        ahold = acquisition.load_label_free_block("A_hold", None)
        _validate_block(ahold, expected="A_hold")
        stage = "form_join_and_seal_A_hold_four_arms"
        ahold_formation = _form_four_arms(
            ahold,
            model,
            planner_runner=planner_runner,
            cross_encoder_scorer=cross_encoder_scorer,
            minilm_encoder=minilm_encoder,
            hippo_runner=hippo_runner,
            gpu0_cache_releaser=gpu0_cache_releaser,
        )
        ahold_rows = ahold_formation.rows
        ahold_action = _seal_json(
            root,
            "A_hold.actions.private.json",
            _four_arm_action_body(
                ahold,
                ahold_rows,
                e1_model_sha256=model_archive.self_sha256,
                gpu0_cache_release_receipt=(
                    ahold_formation.gpu0_cache_release_receipt
                ),
            ),
        )
        stage = "release_A_hold_qrels_after_action_seal"
        ahold_pack = acquisition.release_qrels_after_action_seal(
            "A_hold", ahold_action.path, ahold_action.value
        )
        ahold_qrels = _validate_qrels(
            ahold_pack,
            block=ahold,
            action_archive_sha256=ahold_action.self_sha256,
        )
        ahold_qrel_archive = _seal_qrel_pack(root, ahold_pack)
        stage = "score_A_hold_offline"
        ahold_score = _score_four_arm_block(
            root, ahold, ahold_rows, ahold_qrels
        )

        m_action: _SealedFile | None = None
        m_qrel_archive: _SealedFile | None = None
        m_score: _ScoredBlock | None = None
        if _comparison_pass(ahold_score.comparison_e1_e0):
            stage = "seal_promotion_authorization"
            authorization = _promotion_authorization(
                claim, ahold_score.comparison_e1_e0
            )
            _exclusive_bytes(
                root / PROMOTION_AUTHORIZATION_FILENAME,
                canonical_bytes(authorization),
            )
            stage = "first_decode_and_load_M_search_label_free"
            m_block = acquisition.load_label_free_block(
                "M_search", authorization
            )
            _validate_block(m_block, expected="M_search")
            stage = "form_join_and_seal_M_search_four_arms"
            m_formation = _form_four_arms(
                m_block,
                model,
                planner_runner=planner_runner,
                cross_encoder_scorer=cross_encoder_scorer,
                minilm_encoder=minilm_encoder,
                hippo_runner=hippo_runner,
                gpu0_cache_releaser=gpu0_cache_releaser,
            )
            m_rows = m_formation.rows
            m_action = _seal_json(
                root,
                "M_search.actions.private.json",
                _four_arm_action_body(
                    m_block,
                    m_rows,
                    e1_model_sha256=model_archive.self_sha256,
                    gpu0_cache_release_receipt=(
                        m_formation.gpu0_cache_release_receipt
                    ),
                ),
            )
            stage = "release_M_search_qrels_after_action_seal"
            m_pack = acquisition.release_qrels_after_action_seal(
                "M_search", m_action.path, m_action.value
            )
            m_qrels = _validate_qrels(
                m_pack,
                block=m_block,
                action_archive_sha256=m_action.self_sha256,
            )
            m_qrel_archive = _seal_qrel_pack(root, m_pack)
            stage = "score_M_search_L5_offline"
            m_score = _score_four_arm_block(
                root, m_block, m_rows, m_qrels
            )

        stage = "seal_safe_terminal"
        terminal_body = _terminal_body(
            execution_binding_sha256=binding,
            marker_sha256=marker_sha256,
            claim=claim,
            aform_action=aform_action,
            aform_qrel=aform_qrel_archive,
            model_archive=model_archive,
            ahold_action=ahold_action,
            ahold_qrel=ahold_qrel_archive,
            ahold_score=ahold_score,
            m_action=m_action,
            m_qrel=m_qrel_archive,
            m_score=m_score,
        )
        terminal = self_hashed(terminal_body)
        _exclusive_bytes(
            root / FORMAL_TERMINAL_FILENAME, canonical_bytes(terminal)
        )
        return terminal
    except Exception as exc:
        _write_failure_terminal(
            root,
            execution_binding_sha256=binding,
            marker_sha256=marker_sha256,
            stage=stage,
            exc=exc,
        )
        if isinstance(exc, HitabP1FormalControllerError):
            raise
        raise HitabP1FormalControllerError(
            "formal controller failed closed"
        ) from exc


__all__ = [
    "AcquisitionClaim",
    "BLOCK_COUNTS",
    "BlockView",
    "FAMILIES",
    "FAMILY_COUNTS",
    "FormalAcquisitionBoundary",
    "FormalItemView",
    "GPU0CacheReleaser",
    "HitabP1FormalControllerError",
    "PROMOTION_AUTHORIZATION_FILENAME",
    "QrelPack",
    "QrelRow",
    "run_formal_controller",
    "self_hashed",
    "stable_hash",
]
