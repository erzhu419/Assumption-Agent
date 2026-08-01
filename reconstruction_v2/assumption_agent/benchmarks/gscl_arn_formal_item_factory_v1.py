"""Supervisor-internal four-arm item factory.

The public low-level narrative APIs deliberately accept Python callables and
dataclass instances so that they can be tested.  Those objects are not a
security or provenance boundary and their serialized forms are never accepted
here.  This factory derives its own scorer commitments, constructs
``FrozenNarrativeScorers`` itself, fixes ``MappingSearchConfig`` internally,
immediately runs all four arms, and recomputes the complete result before it
can be sealed by the formal supervisor.

The only formal constructor creates the concrete offline MiniLM encoder from
frozen assets.  The injected-encoder constructor is explicitly source-free
qualification-only; its output carries a different lineage and must be
rejected by the formal supervisor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from assumption_agent.gscl_arn_intrinsic_arms_v1 import (
    IntrinsicArm,
    IntrinsicItemResult,
    evaluate_intrinsic_item,
)
from assumption_agent.gscl_arn_intrinsic_scorers_v1 import (
    FrozenNarrativeScorers,
    LEGACY_FEATURE_IDS,
    LEGACY_REGISTRY_SHA256,
    SCORER_CONTRACT_HASH,
)
from assumption_agent.gscl_narrative_correspondence_v1 import (
    MappingSearchConfig,
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1 import contract as extractor_contract
from replication_runtime.gscl_minilm_portable_v1.binding import (
    GSCLPortableOfflineMiniLMEncoder,
)


VERSION = "gscl_arn_formal_item_factory_v1"
PRIVATE_OUTPUT_SCHEMA = f"{VERSION}.private_four_arm_output.v1"
BATCH_MANIFEST_SCHEMA = f"{VERSION}.extractor_batch_manifest.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = object()
_FORMAL_ARM_IDS = {
    IntrinsicArm.SEMANTIC_ONLY: "semantic_only",
    IntrinsicArm.LEGACY: "legacy_keyword",
    IntrinsicArm.FLAT: "flat_label_no_verifier",
    IntrinsicArm.FULL: "full_gscl",
}
_ARM_ORDER = (
    "semantic_only",
    "legacy_keyword",
    "flat_label_no_verifier",
    "full_gscl",
)
_PREDICTOR_ROW_KEYS = {
    "opaque_item_id",
    "query_narrative",
    "first_choice",
    "second_choice",
}


class FormalItemFactoryError(RuntimeError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _canonical_bytes(value: Any) -> bytes:
    def check(item: Any) -> None:
        if item is None or type(item) in {bool, int, str}:
            return
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise FormalItemFactoryError("canonical_key_invalid")
            for child in item.values():
                check(child)
            return
        raise FormalItemFactoryError("canonical_type_invalid")

    check(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _receipt_json(value: object) -> str:
    if not isinstance(value, Mapping):
        raise FormalItemFactoryError(
            "encoder_binding_receipt_invalid"
        )
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FormalItemFactoryError(
            "encoder_binding_receipt_invalid"
        ) from exc


def _encoder_binding_payload(encoder: object) -> dict[str, str]:
    runtime_json = _receipt_json(
        getattr(encoder, "runtime_receipt", None)
    )
    canary_json = _receipt_json(
        getattr(encoder, "canary_receipt", None)
    )
    return {
        "encoder_exact_type": (
            f"{type(encoder).__module__}.{type(encoder).__qualname__}"
        ),
        "encoder_runtime_receipt_json": runtime_json,
        "encoder_runtime_receipt_sha256": hashlib.sha256(
            runtime_json.encode("ascii")
        ).hexdigest(),
        "encoder_canary_receipt_json": canary_json,
        "encoder_canary_receipt_sha256": hashlib.sha256(
            canary_json.encode("ascii")
        ).hexdigest(),
    }


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FormalItemFactoryError("factory_module_unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scorer_state_commitment(scorers: FrozenNarrativeScorers) -> str:
    """Bind the constructor-verified immutable state without re-copying it."""

    scorers.validate_internal()
    return _hash(
        {
            "source_vector_commitment": scorers.receipt[
                "source_vector_commitment"
            ],
            "mention_vector_commitment": scorers.receipt[
                "mention_vector_commitment"
            ],
            "extraction_set_commitment": scorers.receipt[
                "extraction_set_commitment"
            ],
            "source_count": scorers.receipt["source_count"],
            "mention_count": scorers.receipt["mention_count"],
            "extraction_count": scorers.receipt["extraction_count"],
        }
    )


def _formal_encoder_from_assets(
    *,
    asset_manifest_path: Path,
    model_root: Path,
    target_manifest_path: Path,
) -> GSCLPortableOfflineMiniLMEncoder:
    return GSCLPortableOfflineMiniLMEncoder(
        asset_manifest_path=asset_manifest_path,
        model_root=model_root,
        target_manifest_path=target_manifest_path,
        run_canary=True,
    )


@dataclass(frozen=True)
class PrivateFactoryItemOutput:
    """Token-bound in-process output; item-level content remains private."""

    opaque_item_id: str
    prediction_rows: tuple[Mapping[str, Any], ...]
    recomputation_receipt: Mapping[str, Any]
    lineage: str
    _token: object


@dataclass(frozen=True)
class FrozenArnItemFactory:
    """Concrete scorer factory over one already-grounded extraction set."""

    scorers: FrozenNarrativeScorers
    extraction_hashes: frozenset[str]
    lineage: str
    factory_receipt: Mapping[str, Any]
    _token: object

    @classmethod
    def from_frozen_assets(
        cls,
        *,
        extractions: Sequence[NarrativeExtraction],
        asset_manifest_path: Path,
        model_root: Path,
        target_manifest_path: Path,
    ) -> "FrozenArnItemFactory":
        """The only formal constructor; no encoder or commitments are injected."""

        encoder = _formal_encoder_from_assets(
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
            target_manifest_path=target_manifest_path,
        )
        return cls._build(
            extractions=extractions,
            encoder=encoder,
            lineage="formal_frozen_assets",
        )

    @classmethod
    def _source_free_qualification(
        cls,
        *,
        extractions: Sequence[NarrativeExtraction],
        encoder: object,
    ) -> "FrozenArnItemFactory":
        """Qualification-only constructor; never accepted as formal evidence."""

        return cls._build(
            extractions=extractions,
            encoder=encoder,
            lineage="synthetic_source_free_qualification",
        )

    @classmethod
    def _build(
        cls,
        *,
        extractions: Sequence[NarrativeExtraction],
        encoder: object,
        lineage: str,
    ) -> "FrozenArnItemFactory":
        if cls is not FrozenArnItemFactory:
            raise FormalItemFactoryError("factory_subclass_forbidden")
        rows = tuple(extractions)
        if (
            not rows
            or any(not isinstance(row, NarrativeExtraction) for row in rows)
            or len({row.extraction_hash for row in rows}) != len(rows)
        ):
            raise FormalItemFactoryError("factory_extractions_invalid")
        for row in rows:
            row.__post_init__()
        scorers = FrozenNarrativeScorers.build(rows, encoder=encoder)
        is_formal_encoder = (
            type(encoder) is GSCLPortableOfflineMiniLMEncoder
        )
        if (
            lineage == "formal_frozen_assets"
            and (
                not is_formal_encoder
                or scorers.receipt.get("construction_domain")
                != (
                    "formal_exact_gscl_target_local_"
                    "portable_minilm_v1"
                )
            )
        ) or (
            lineage == "synthetic_source_free_qualification"
            and is_formal_encoder
        ):
            raise FormalItemFactoryError(
                "factory_encoder_lineage_invalid"
            )
        if type(scorers) is not FrozenNarrativeScorers:
            raise FormalItemFactoryError("scorer_subclass_forbidden")
        # ``FrozenNarrativeScorers.build`` already copies every vector into a
        # bytes-backed ndarray and seals both mappings.  Reconstructing the
        # dataclass here would bypass its factory-only authority boundary.
        scorers.validate_internal()
        source_files = {
            "factory": _file_hash(Path(__file__)),
            "arms": _file_hash(
                Path(
                    __import__(
                        "assumption_agent.gscl_arn_intrinsic_arms_v1",
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
            "scorers": _file_hash(
                Path(
                    __import__(
                        "assumption_agent.gscl_arn_intrinsic_scorers_v1",
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
            "narrative_core": _file_hash(
                Path(
                    __import__(
                        "assumption_agent.gscl_narrative_correspondence_v1",
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
            "gscl_minilm_binding": _file_hash(
                Path(
                    __import__(
                        (
                            "replication_runtime."
                            "gscl_minilm_portable_v1.binding"
                        ),
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
            "portable_minilm_binding": _file_hash(
                Path(
                    __import__(
                        (
                            "replication_runtime."
                            "qasper_minilm_portable_v2.binding"
                        ),
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
            "base_minilm_binding": _file_hash(
                Path(
                    __import__(
                        "replication_runtime.qasper_minilm_v1.binding",
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
        }
        encoder_binding = _encoder_binding_payload(encoder)
        receipt: dict[str, Any] = {
            "version": VERSION,
            "lineage": lineage,
            "scorer_contract_hash": SCORER_CONTRACT_HASH,
            "scorer_receipt_self_hash": scorers.receipt["self_hash"],
            "scorer_state_commitment": _scorer_state_commitment(scorers),
            "legacy_registry_sha256": LEGACY_REGISTRY_SHA256,
            "mapping_config": MappingSearchConfig().safe_payload(),
            "source_file_sha256s": dict(sorted(source_files.items())),
            "extraction_set_commitment": _hash(
                sorted(row.extraction_hash for row in rows)
            ),
            **encoder_binding,
            "caller_supplied_commitments_accepted": False,
            "caller_supplied_predictions_accepted": False,
            "caller_supplied_prepared_results_accepted": False,
            "same_process_objects_are_security_boundary": False,
            "formal_security_boundary": (
                "closure_bound_supervisor_plus_landlock_child"
            ),
        }
        receipt["self_hash"] = _hash(receipt)
        return cls(
            scorers=scorers,
            extraction_hashes=frozenset(
                row.extraction_hash for row in rows
            ),
            lineage=lineage,
            factory_receipt=MappingProxyType(receipt),
            _token=_TOKEN,
        )

    def _commitment(self, component: str) -> str:
        if component not in {
            "raw_text_scorer",
            "legacy_vectorizer",
            "structural_scorer",
        }:
            raise FormalItemFactoryError("factory_component_invalid")
        return _hash(
            {
                "component": component,
                "factory_receipt_self_hash": self.factory_receipt[
                    "self_hash"
                ],
                "scorer_contract_hash": SCORER_CONTRACT_HASH,
                "scorer_receipt_self_hash": self.scorers.receipt[
                    "self_hash"
                ],
            }
        )

    def _validate_factory_state(self) -> None:
        body = dict(self.factory_receipt)
        claimed = body.pop("self_hash", None)
        scorer_body = dict(self.scorers.receipt)
        scorer_claimed = scorer_body.pop("self_hash", None)
        if (
            type(self) is not FrozenArnItemFactory
            or type(self.scorers) is not FrozenNarrativeScorers
            or self._token is not _TOKEN
            or not isinstance(claimed, str)
            or _hash(body) != claimed
            or not isinstance(scorer_claimed, str)
            or _hash(scorer_body) != scorer_claimed
            or scorer_claimed
            != self.factory_receipt["scorer_receipt_self_hash"]
            or _scorer_state_commitment(self.scorers)
            != self.factory_receipt["scorer_state_commitment"]
        ):
            raise FormalItemFactoryError("factory_state_changed")
        for component, expected in self.factory_receipt[
            "source_file_sha256s"
        ].items():
            if component == "factory":
                path = Path(__file__)
            else:
                module_name = {
                    "arms": "assumption_agent.gscl_arn_intrinsic_arms_v1",
                    "base_minilm_binding": (
                        "replication_runtime.qasper_minilm_v1.binding"
                    ),
                    "gscl_minilm_binding": (
                        "replication_runtime."
                        "gscl_minilm_portable_v1.binding"
                    ),
                    "portable_minilm_binding": (
                        "replication_runtime."
                        "qasper_minilm_portable_v2.binding"
                    ),
                    "scorers": (
                        "assumption_agent.gscl_arn_intrinsic_scorers_v1"
                    ),
                    "narrative_core": (
                        "assumption_agent.gscl_narrative_correspondence_v1"
                    ),
                }[component]
                path = Path(
                    __import__(module_name, fromlist=["__file__"]).__file__
                )
            if _file_hash(path) != expected:
                raise FormalItemFactoryError(
                    "factory_implementation_changed"
                )

    def evaluate_private_item(
        self,
        *,
        opaque_item_id: str,
        query: NarrativeExtraction,
        candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    ) -> PrivateFactoryItemOutput:
        """Generate and immediately choose all arms, then recompute exactly."""

        self._validate_factory_state()
        rows = (query, *candidates)
        if (
            any(not isinstance(row, NarrativeExtraction) for row in rows)
            or any(
                row.extraction_hash not in self.extraction_hashes
                for row in rows
            )
        ):
            raise FormalItemFactoryError("item_extraction_not_primed")
        for row in rows:
            row.__post_init__()
        config = MappingSearchConfig()
        commitments = {
            "raw_text_scorer_commitment": self._commitment(
                "raw_text_scorer"
            ),
            "legacy_vectorizer_commitment": self._commitment(
                "legacy_vectorizer"
            ),
            "structural_scorer_commitment": self._commitment(
                "structural_scorer"
            ),
        }

        def compute() -> IntrinsicItemResult:
            return evaluate_intrinsic_item(
                opaque_item_id=opaque_item_id,
                query=query,
                candidates=candidates,
                raw_text_scorer=self.scorers.raw_text_scorer,
                legacy_vectorizer=self.scorers.legacy_vectorizer,
                legacy_feature_ids=LEGACY_FEATURE_IDS,
                structural_scorer=self.scorers.structural_scorer,
                mapping_config=config,
                **commitments,
            )

        first = compute()
        second = compute()
        first.__post_init__()
        second.__post_init__()
        first_payload = first.safe_payload()
        second_payload = second.safe_payload()
        if (
            first_payload != second_payload
            or first.result_hash != second.result_hash
        ):
            raise FormalItemFactoryError("item_recomputation_drifted")
        self._deep_validate_result(
            first,
            opaque_item_id=opaque_item_id,
            query=query,
            candidates=candidates,
            commitments=commitments,
            config=config,
        )
        prediction_rows = tuple(
            {
                "arm_id": _FORMAL_ARM_IDS[prediction.arm],
                "disposition": prediction.disposition.value,
                "predicted_ordinal": prediction.predicted_ordinal,
                "evidence_commitment": prediction.evidence_commitment,
                "reason_ids": list(prediction.reason_ids),
            }
            for prediction in first.predictions
        )
        receipt: dict[str, Any] = {
            "version": VERSION,
            "factory_receipt_self_hash": self.factory_receipt["self_hash"],
            "opaque_item_id": opaque_item_id,
            "result_hash": first.result_hash,
            "second_result_hash": second.result_hash,
            "deep_recomputation_exact": True,
            "prediction_set_commitment": _hash(
                [dict(row) for row in prediction_rows]
            ),
            "item_content_emitted": False,
        }
        receipt["self_hash"] = _hash(receipt)
        return PrivateFactoryItemOutput(
            opaque_item_id=opaque_item_id,
            prediction_rows=prediction_rows,
            recomputation_receipt=receipt,
            lineage=self.lineage,
            _token=_TOKEN,
        )

    def _deep_validate_result(
        self,
        result: IntrinsicItemResult,
        *,
        opaque_item_id: str,
        query: NarrativeExtraction,
        candidates: tuple[NarrativeExtraction, NarrativeExtraction],
        commitments: Mapping[str, str],
        config: MappingSearchConfig,
    ) -> None:
        expected_arms = tuple(IntrinsicArm)
        implementation = dict(result.implementation_commitments)
        if (
            result.opaque_item_id != opaque_item_id
            or result.query_extraction_hash != query.extraction_hash
            or result.query_provenance_hash != query.provenance_hash
            or result.candidate_extraction_hashes
            != tuple(row.extraction_hash for row in candidates)
            or result.candidate_provenance_hashes
            != tuple(row.provenance_hash for row in candidates)
            or tuple(row.arm for row in result.predictions)
            != expected_arms
            or len(result.candidate_receipts) != 2
            or implementation.get("raw_text_scorer")
            != commitments["raw_text_scorer_commitment"]
            or implementation.get("legacy_vectorizer")
            != commitments["legacy_vectorizer_commitment"]
            or implementation.get("structural_scorer")
            != commitments["structural_scorer_commitment"]
            or implementation.get("legacy_registry")
            != _hash(list(LEGACY_FEATURE_IDS))
            or implementation.get("mapping_config") != config.config_hash
        ):
            raise FormalItemFactoryError(
                "internal_result_cross_binding_invalid"
            )
        common_input_commitments = {
            row.input_commitment for row in result.predictions
        }
        if len(common_input_commitments) != 1:
            raise FormalItemFactoryError(
                "internal_prediction_input_binding_invalid"
            )
        for ordinal, receipt in enumerate(result.candidate_receipts):
            if (
                receipt.candidate_ordinal != ordinal
                or receipt.candidate_extraction_hash
                != candidates[ordinal].extraction_hash
                or receipt.candidate_provenance_hash
                != candidates[ordinal].provenance_hash
                or receipt.flat_proposal_set_hash
                != receipt.full_proposal_set_hash
            ):
                raise FormalItemFactoryError(
                    "internal_candidate_binding_invalid"
                )


def _decode_canonical_object(raw: bytes, *, issue_id: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=unique,
            parse_float=lambda _: (_ for _ in ()).throw(ValueError()),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalItemFactoryError(issue_id) from exc
    if (
        not isinstance(value, dict)
        or _canonical_bytes(value) + b"\n" != raw
    ):
        raise FormalItemFactoryError(issue_id)
    return value


def _predictor_rows(raw: bytes) -> tuple[dict[str, str], ...]:
    pack = _decode_canonical_object(raw, issue_id="predictor_pack_invalid")
    rows = pack.get("rows")
    if not isinstance(rows, list) or not rows:
        raise FormalItemFactoryError("predictor_pack_invalid")
    checked: list[dict[str, str]] = []
    observed: set[str] = set()
    for value in rows:
        if not isinstance(value, dict) or set(value) != _PREDICTOR_ROW_KEYS:
            raise FormalItemFactoryError("predictor_row_invalid")
        opaque = value["opaque_item_id"]
        if (
            not isinstance(opaque, str)
            or _SHA256.fullmatch(opaque) is None
            or opaque in observed
        ):
            raise FormalItemFactoryError("predictor_item_id_invalid")
        row: dict[str, str] = {"opaque_item_id": opaque}
        for field in (
            "query_narrative",
            "first_choice",
            "second_choice",
        ):
            text = value[field]
            if (
                not isinstance(text, str)
                or not text
                or "\x00" in text
                or len(text.encode("utf-8")) > 128 * 1024
            ):
                raise FormalItemFactoryError("predictor_story_invalid")
            row[field] = text
        if row["first_choice"] == row["second_choice"]:
            raise FormalItemFactoryError("predictor_choices_identical")
        checked.append(row)
        observed.add(opaque)
    if checked != sorted(checked, key=lambda row: row["opaque_item_id"]):
        raise FormalItemFactoryError("predictor_order_invalid")
    return tuple(checked)


def _expected_stories(
    rows: Sequence[Mapping[str, str]],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (row["opaque_item_id"], role, row[field])
        for row in rows
        for role, field in (
            ("query", "query_narrative"),
            ("first_choice", "first_choice"),
            ("second_choice", "second_choice"),
        )
    )


def _decode_extractor_batches(
    *,
    predictor_rows: Sequence[Mapping[str, str]],
    batches: Sequence[tuple[object, ...]],
) -> tuple[
    dict[str, dict[str, NarrativeExtraction]],
    set[str],
    list[dict[str, Any]],
]:
    expected = _expected_stories(predictor_rows)
    expected_triplets = tuple(
        tuple(expected[offset : offset + 3])
        for offset in range(0, len(expected), 3)
    )
    legacy_cursor = 0
    seen_item_indices: set[int] = set()
    extractions: dict[str, dict[str, NarrativeExtraction]] = {}
    invalid_items: set[str] = set()
    receipts: list[dict[str, Any]] = []
    closure: Mapping[str, object] | None = None
    for sequence, raw_batch in enumerate(batches):
        if not isinstance(raw_batch, tuple) or len(raw_batch) not in {
            3,
            4,
        }:
            raise FormalItemFactoryError(
                "extractor_batch_binding_invalid"
            )
        pack, output, output_sha256 = raw_batch[:3]
        if (
            not isinstance(
                pack, extractor_contract.StoryOnlyInputPack
            )
            or not isinstance(output, Mapping)
            or not isinstance(output_sha256, str)
            or _SHA256.fullmatch(output_sha256) is None
        ):
            raise FormalItemFactoryError(
                "extractor_batch_binding_invalid"
            )
        if len(raw_batch) == 4:
            raw_indices = raw_batch[3]
            if (
                not isinstance(raw_indices, tuple)
                or not raw_indices
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < len(expected_triplets)
                    for value in raw_indices
                )
                or tuple(sorted(raw_indices)) != raw_indices
                or len(set(raw_indices)) != len(raw_indices)
            ):
                raise FormalItemFactoryError(
                    "extractor_batch_item_indices_invalid"
                )
            item_indices = raw_indices
        else:
            if len(pack.requests) % 3 != 0:
                raise FormalItemFactoryError(
                    "extractor_batch_item_indices_invalid"
                )
            item_count = len(pack.requests) // 3
            item_indices = tuple(
                range(
                    legacy_cursor,
                    legacy_cursor + item_count,
                )
            )
            legacy_cursor += item_count
        if (
            seen_item_indices.intersection(item_indices)
            or len(pack.requests) != 3 * len(item_indices)
        ):
            raise FormalItemFactoryError(
                "extractor_batch_item_indices_invalid"
            )
        seen_item_indices.update(item_indices)
        batch_expected = tuple(
            row
            for item_index in item_indices
            for row in expected_triplets[item_index]
        )
        if pack.sequence != sequence or output["sequence"] != sequence:
            raise FormalItemFactoryError("extractor_batch_sequence_invalid")
        if closure is None:
            closure = output["execution_closure"]
        elif output["execution_closure"] != closure:
            raise FormalItemFactoryError(
                "extractor_execution_closure_changed"
            )
        results = output["results"]
        if not isinstance(results, list) or len(results) != len(pack.requests):
            raise FormalItemFactoryError("extractor_result_count_invalid")
        for position, (request, result) in enumerate(
            zip(pack.requests, results, strict=True)
        ):
            opaque, role, story = batch_expected[position]
            if request.story_text != story or request.ordinal != position:
                raise FormalItemFactoryError(
                    "extractor_story_binding_invalid"
                )
            if not isinstance(result, dict):
                raise FormalItemFactoryError("extractor_result_invalid")
            if result["generation_valid"] is not True:
                invalid_items.add(opaque)
                continue
            completion = result["completion"]
            try:
                extraction = parse_untrusted_generator_completion(
                    NarrativeSource(
                        "private."
                        + hashlib.sha256(
                            (
                                pack.story_commitments[position]
                                + role
                            ).encode("ascii")
                        ).hexdigest()[:32],
                        story,
                    ),
                    completion,
                )
            except Exception:
                invalid_items.add(opaque)
                continue
            extractions.setdefault(opaque, {})[role] = extraction
        receipts.append(
            {
                "sequence": sequence,
                "batch_id": pack.batch_id,
                "input_file_sha256": pack.input_file_sha256,
                "input_pack_commitment": pack.input_pack_commitment,
                "output_file_sha256": output_sha256,
                "execution_closure_commitment": _hash(
                    output["execution_closure"]
                ),
                "story_count": len(pack.requests),
            }
        )
    if seen_item_indices != set(range(len(expected_triplets))):
        raise FormalItemFactoryError("extractor_story_count_invalid")
    for row in predictor_rows:
        opaque = row["opaque_item_id"]
        if set(extractions.get(opaque, {})) != {
            "query",
            "first_choice",
            "second_choice",
        }:
            invalid_items.add(opaque)
    return extractions, invalid_items, receipts


def _error_prediction(opaque_item_id: str) -> dict[str, Any]:
    return {
        "opaque_item_id": opaque_item_id,
        "disposition": "ERROR",
        "selected_choice": None,
        "error_code": "ARM_RUNTIME_ERROR",
    }


def _factory_prediction(
    opaque_item_id: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    if row["disposition"] == "predicted":
        ordinal = row["predicted_ordinal"]
        if ordinal not in {0, 1}:
            raise FormalItemFactoryError("factory_prediction_invalid")
        return {
            "opaque_item_id": opaque_item_id,
            "disposition": "ANSWER",
            "selected_choice": (
                "first_choice" if ordinal == 0 else "second_choice"
            ),
            "error_code": None,
        }
    if row["disposition"] == "abstain":
        return {
            "opaque_item_id": opaque_item_id,
            "disposition": "ABSTAIN",
            "selected_choice": None,
            "error_code": None,
        }
    raise FormalItemFactoryError("factory_prediction_invalid")


def _build_private_four_arm_output(
    *,
    predictor_raw: bytes,
    batches: Sequence[tuple[object, ...]],
    asset_manifest_path: Path | None,
    model_root: Path | None,
    target_manifest_path: Path | None,
    qualification_encoder: object | None,
) -> dict[str, Any]:
    rows = _predictor_rows(predictor_raw)
    extractions, invalid_items, batch_receipts = _decode_extractor_batches(
        predictor_rows=rows, batches=batches
    )
    valid_extractions = tuple(
        extraction
        for row in rows
        if row["opaque_item_id"] not in invalid_items
        for extraction in (
            extractions[row["opaque_item_id"]]["query"],
            extractions[row["opaque_item_id"]]["first_choice"],
            extractions[row["opaque_item_id"]]["second_choice"],
        )
    )
    standalone_encoder_binding: dict[str, str] | None = None
    if qualification_encoder is None:
        if (
            asset_manifest_path is None
            or model_root is None
            or target_manifest_path is None
        ):
            raise FormalItemFactoryError("formal_factory_assets_invalid")
        if valid_extractions:
            factory = FrozenArnItemFactory.from_frozen_assets(
                extractions=valid_extractions,
                asset_manifest_path=asset_manifest_path,
                model_root=model_root,
                target_manifest_path=target_manifest_path,
            )
        else:
            encoder = _formal_encoder_from_assets(
                asset_manifest_path=asset_manifest_path,
                model_root=model_root,
                target_manifest_path=target_manifest_path,
            )
            standalone_encoder_binding = _encoder_binding_payload(
                encoder
            )
            factory = None
        lineage = "formal_frozen_assets"
    else:
        if (
            asset_manifest_path is not None
            or model_root is not None
            or target_manifest_path is not None
        ):
            raise FormalItemFactoryError(
                "qualification_factory_assets_ambiguous"
            )
        factory = (
            FrozenArnItemFactory._source_free_qualification(
                extractions=valid_extractions,
                encoder=qualification_encoder,
            )
            if valid_extractions
            else None
        )
        lineage = "synthetic_source_free_qualification"
    by_arm: dict[str, list[dict[str, Any]]] = {
        arm_id: [] for arm_id in _ARM_ORDER
    }
    private_item_receipts: list[dict[str, str]] = []
    for row in rows:
        opaque = row["opaque_item_id"]
        if opaque in invalid_items or factory is None:
            for arm_id in _ARM_ORDER:
                by_arm[arm_id].append(_error_prediction(opaque))
            continue
        item = extractions[opaque]
        output = factory.evaluate_private_item(
            opaque_item_id=opaque,
            query=item["query"],
            candidates=(
                item["first_choice"],
                item["second_choice"],
            ),
        )
        if output.lineage != lineage:
            raise FormalItemFactoryError("factory_lineage_invalid")
        indexed = {
            prediction["arm_id"]: prediction
            for prediction in output.prediction_rows
        }
        if set(indexed) != set(_ARM_ORDER):
            raise FormalItemFactoryError("factory_arm_set_invalid")
        for arm_id in _ARM_ORDER:
            by_arm[arm_id].append(
                _factory_prediction(opaque, indexed[arm_id])
            )
        private_item_receipts.append(
            {
                "opaque_item_id": opaque,
                "recomputation_receipt_self_hash": (
                    output.recomputation_receipt["self_hash"]
                ),
            }
        )
    body: dict[str, Any] = {
        "schema": PRIVATE_OUTPUT_SCHEMA,
        "status": "PRIVATE_ALL_FOUR_ITEM_RESULTS_RECOMPUTED",
        "lineage": lineage,
        "predictor_pack_file_sha256": hashlib.sha256(
            predictor_raw
        ).hexdigest(),
        "extractor_batch_receipts": batch_receipts,
        "factory_receipt_self_hash": (
            None if factory is None else factory.factory_receipt["self_hash"]
        ),
        "encoder_binding": (
            standalone_encoder_binding
            if factory is None
            else {
                key: factory.factory_receipt[key]
                for key in (
                    "encoder_exact_type",
                    "encoder_runtime_receipt_json",
                    "encoder_runtime_receipt_sha256",
                    "encoder_canary_receipt_json",
                    "encoder_canary_receipt_sha256",
                )
            }
        ),
        "by_arm": by_arm,
        "private_item_recomputation_receipts": private_item_receipts,
        "item_count": len(rows),
        "error_item_count": len(invalid_items),
        "caller_predictions_accepted": False,
        "caller_commitments_accepted": False,
        "item_content_emitted": False,
    }
    body["self_hash"] = _hash(body)
    return body


def build_private_four_arm_output_qualification_only(
    *,
    predictor_raw: bytes,
    input_batch_raws: Sequence[bytes],
    output_batch_raws: Sequence[bytes],
    encoder: object,
) -> dict[str, Any]:
    """Source-free mechanical bridge; its lineage cannot become formal."""

    if len(input_batch_raws) != len(output_batch_raws):
        raise FormalItemFactoryError("qualification_batch_count_invalid")
    batches = []
    for input_raw, output_raw in zip(
        input_batch_raws, output_batch_raws, strict=True
    ):
        pack = extractor_contract.admit_story_only_pack_qualification_only(
            input_raw
        )
        output = extractor_contract.decode_private_output(
            output_raw, expected_pack=pack
        )
        batches.append(
            (pack, output, hashlib.sha256(output_raw).hexdigest())
        )
    return _build_private_four_arm_output(
        predictor_raw=predictor_raw,
        batches=batches,
        asset_manifest_path=None,
        model_root=None,
        target_manifest_path=None,
        qualification_encoder=encoder,
    )


def _write_private_output_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(dict(value)) + b"\n"
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise FormalItemFactoryError(
            "factory_output_path_invalid"
        ) from exc
    if (
        not path.is_absolute()
        or absolute != path
        or absolute.name in {"", ".", ".."}
    ):
        raise FormalItemFactoryError("factory_output_parent_invalid")
    try:
        _, parent_descriptor = (
            extractor_contract._open_trusted_directory(  # noqa: SLF001
                absolute.parent,
                final_mode=0o700,
                final_owner_current=True,
            )
        )
    except extractor_contract.NarrativeExtractorRuntimeError as exc:
        raise FormalItemFactoryError(
            "factory_output_parent_invalid"
        ) from exc
    descriptor: int | None = None
    try:
        descriptor = os.open(
            absolute.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise FormalItemFactoryError(
                    "factory_output_write_failed"
                )
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(raw)
        ):
            raise FormalItemFactoryError(
                "factory_output_topology_invalid"
            )
        os.fsync(parent_descriptor)
    except FileExistsError as exc:
        raise FormalItemFactoryError(
            "factory_output_already_exists"
        ) from exc
    except OSError as exc:
        raise FormalItemFactoryError(
            "factory_output_write_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def run_formal_factory_files(
    *,
    predictor_path: Path,
    batch_manifest_path: Path,
    minilm_manifest_path: Path,
    minilm_model_root: Path,
    minilm_target_manifest_path: Path,
    output_path: Path,
) -> None:
    """Secure formal file bridge invoked only by the Landlocked supervisor."""

    predictor_read = extractor_contract.secure_read_file(
        predictor_path, maximum=20 * 1024 * 1024
    )
    manifest_read = extractor_contract.secure_read_file(
        batch_manifest_path, maximum=2 * 1024 * 1024
    )
    manifest = _decode_canonical_object(
        manifest_read.raw, issue_id="factory_batch_manifest_invalid"
    )
    if (
        set(manifest)
        != {
            "schema",
            "predictor_pack_file_sha256",
            "batches",
        }
        or manifest["schema"] != BATCH_MANIFEST_SCHEMA
        or manifest["predictor_pack_file_sha256"]
        != predictor_read.sha256
        or not isinstance(manifest["batches"], list)
        or not manifest["batches"]
    ):
        raise FormalItemFactoryError("factory_batch_manifest_invalid")
    batches = []
    for sequence, row in enumerate(manifest["batches"]):
        if not isinstance(row, dict) or set(row) != {
            "sequence",
            "item_indices",
            "input_path",
            "input_file_sha256",
            "output_path",
            "output_file_sha256",
        }:
            raise FormalItemFactoryError("factory_batch_manifest_invalid")
        item_indices = row["item_indices"]
        if (
            row["sequence"] != sequence
            or not isinstance(item_indices, list)
            or not item_indices
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in item_indices
            )
            or item_indices != sorted(item_indices)
            or len(set(item_indices)) != len(item_indices)
        ):
            raise FormalItemFactoryError("factory_batch_sequence_invalid")
        input_path = Path(row["input_path"])
        extractor_output_path = Path(row["output_path"])
        pack = extractor_contract.load_trusted_story_only_input_pack(
            input_path
        )
        output_read = extractor_contract.secure_read_file(
            extractor_output_path,
            maximum=extractor_contract.MAXIMUM_OUTPUT_BYTES,
        )
        if (
            pack.input_file_sha256 != row["input_file_sha256"]
            or output_read.sha256 != row["output_file_sha256"]
        ):
            raise FormalItemFactoryError(
                "factory_batch_file_hash_invalid"
            )
        output = extractor_contract.decode_private_output(
            output_read.raw, expected_pack=pack
        )
        batches.append(
            (pack, output, output_read.sha256, tuple(item_indices))
        )
    result = _build_private_four_arm_output(
        predictor_raw=predictor_read.raw,
        batches=batches,
        asset_manifest_path=minilm_manifest_path,
        model_root=minilm_model_root,
        target_manifest_path=minilm_target_manifest_path,
        qualification_encoder=None,
    )
    if result["lineage"] != "formal_frozen_assets":
        raise FormalItemFactoryError("formal_factory_lineage_invalid")
    _write_private_output_once(output_path, result)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", required=True, type=Path)
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--minilm-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument(
        "--minilm-target-manifest", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run_formal_factory_files(
        predictor_path=arguments.predictor,
        batch_manifest_path=arguments.batch_manifest,
        minilm_manifest_path=arguments.minilm_manifest,
        minilm_model_root=arguments.minilm_model,
        minilm_target_manifest_path=(
            arguments.minilm_target_manifest
        ),
        output_path=arguments.output,
    )
    return 0


def require_internal_formal_output(
    output: object,
) -> PrivateFactoryItemOutput:
    """Accept token-bound outputs from the formal-assets lineage only."""

    if (
        not isinstance(output, PrivateFactoryItemOutput)
        or output._token is not _TOKEN
        or output.lineage != "formal_frozen_assets"
    ):
        raise FormalItemFactoryError("external_or_qualification_output_rejected")
    return output


__all__ = [
    "BATCH_MANIFEST_SCHEMA",
    "FormalItemFactoryError",
    "FrozenArnItemFactory",
    "PRIVATE_OUTPUT_SCHEMA",
    "PrivateFactoryItemOutput",
    "VERSION",
    "build_private_four_arm_output_qualification_only",
    "require_internal_formal_output",
    "run_formal_factory_files",
]


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except FormalItemFactoryError as exc:
        print(
            f"gscl_arn_formal_item_factory_v1 failed closed: {exc.issue_id}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
