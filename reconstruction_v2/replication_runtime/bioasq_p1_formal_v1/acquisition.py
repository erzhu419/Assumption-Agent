"""One-shot sealed-source acquisition boundary for formal BioASQ P1.

The boundary is the only runtime object allowed to open source-compiler
artifacts.  Before actions it releases only the common ordinal/text corpus and
label-free ``work_id``/question projections.  It opens a private qrel pack only
after the controller supplies the exact durable mode-0400 action archive for
that block.  ``M_search`` additionally requires the exact persisted promotion
authorization emitted by the frozen controller.

There is no retry path: each public artifact and each late qrel pack has a
single attempted-open lifecycle.  Content-bearing source artifacts are never
copied, logged, or written by this module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Any
import unicodedata

from assumption_agent.benchmarks import (
    bioasq_p1_formal_controller_v1 as ctl,
)
from assumption_agent.benchmarks import bioasq_p1_formal_source_v2 as source
from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core
from replication_runtime.bioasq_p1_formal_v1.contract import (
    BioasqP1FormalRuntimeError,
    assert_no_symlink_components,
    canonical_bytes,
    required_sha256,
    stable_hash,
    verify_self_hash,
)


VERSION = "bioasq_p1_formal_acquisition_v1"
STUDY_ID = ctl.STUDY_ID
_READ_CHUNK_BYTES = 1 << 20

_SELECTION_RECEIPT_KEYS = frozenset(
    {
        "artifact_binding",
        "compiler_boundary",
        "corpus_aggregate",
        "disjointness_aggregate",
        "p0_binding",
        "quota",
        "schema",
        "seal_contract",
        "selection",
        "self_sha256",
        "source_access",
        "status",
        "study_id",
        "typed_core_binding",
        "version",
    }
)
_ARTIFACT_BINDING_KEYS = frozenset(
    {
        "private_qrels",
        "private_selection_secret",
        "public_blocks",
        "public_corpus",
    }
)
_FILE_BINDING_KEYS = frozenset(
    {"file_sha256", "mode", "row_count", "self_sha256", "size_bytes"}
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "A_hold_E1_minus_E0",
        "block_disjointness_commitment",
        "comparison_net_strictly_positive",
        "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth",
        "schema",
        "self_sha256",
        "status",
        "study_id",
    }
)
_COMPARISON_KEYS = frozenset(
    {
        "item_count",
        "negative_count",
        "net_utility",
        "one_sided_exact_magnitude_preserving_tail",
        "positive_count",
        "tie_count",
    }
)
_FRACTION_KEYS = frozenset({"denominator", "numerator"})
_MARKER_KEYS = frozenset(
    {
        "execution_binding_sha256",
        ctl.NO_CHANGE_COUNT_KEY,
        "schema",
        "self_sha256",
        "study_id",
    }
)


def _absolute(path: Path, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise BioasqP1FormalRuntimeError(f"{field} must be absolute")
    return path


def _direct_file_bytes(
    path: Path,
    *,
    mode: int,
    field: str,
) -> bytes:
    """Read one direct regular file through one no-follow descriptor."""

    checked = _absolute(path, field)
    assert_no_symlink_components(checked, field)
    try:
        before = checked.lstat()
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(f"{field} is unavailable") from exc
    if (
        checked.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != mode
    ):
        raise BioasqP1FormalRuntimeError(
            f"{field} is not a direct mode-{mode:04o} regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(checked, flags)
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            stat.S_IMODE(opened.st_mode),
            opened.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            mode,
            1,
        ):
            raise BioasqP1FormalRuntimeError(
                f"{field} changed during its sole open"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise BioasqP1FormalRuntimeError(
                f"{field} changed during its sole read"
            )
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(f"{field} cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise BioasqP1FormalRuntimeError(f"{field} size changed")
    return raw


def _decode_canonical_mapping(
    raw: bytes,
    *,
    newline: bool,
    field: str,
) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BioasqP1FormalRuntimeError(
            f"{field} is not ASCII JSON"
        ) from exc
    if not isinstance(value, dict):
        raise BioasqP1FormalRuntimeError(f"{field} is not a JSON object")
    if raw != canonical_bytes(value, newline=newline):
        raise BioasqP1FormalRuntimeError(f"{field} is not canonical JSON")
    verify_self_hash(value, field)
    return value


def _read_canonical_mapping(
    path: Path,
    *,
    mode: int,
    newline: bool,
    field: str,
) -> tuple[dict[str, object], bytes]:
    raw = _direct_file_bytes(path, mode=mode, field=field)
    return (
        _decode_canonical_mapping(
            raw,
            newline=newline,
            field=field,
        ),
        raw,
    )


def _validate_file_binding(
    value: object,
    *,
    expected_mode: int,
    expected_rows: int,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FILE_BINDING_KEYS:
        raise BioasqP1FormalRuntimeError(f"{field} binding schema drifted")
    binding = dict(value)
    required_sha256(binding.get("file_sha256"), f"{field} file")
    required_sha256(binding.get("self_sha256"), f"{field} self")
    if (
        binding.get("mode") != f"{expected_mode:04o}"
        or binding.get("row_count") != expected_rows
        or type(binding.get("size_bytes")) is not int
        or binding["size_bytes"] <= 0
    ):
        raise BioasqP1FormalRuntimeError(f"{field} binding drifted")
    return binding


def _query_group_key(question_text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", question_text).casefold().split()
    )


class SealedSourceAcquisitionBoundary:
    """Exact source-artifact loader implementing the controller protocol."""

    def __init__(
        self,
        *,
        outputs: source.FormalOutputPaths,
        selection_receipt: Mapping[str, object],
        controller_root: Path,
        hippo_lane: object,
    ) -> None:
        if not isinstance(outputs, source.FormalOutputPaths):
            raise BioasqP1FormalRuntimeError(
                "formal source output registry drifted"
            )
        if not isinstance(selection_receipt, Mapping):
            raise BioasqP1FormalRuntimeError(
                "selection receipt is unavailable"
            )
        paths = tuple(
            _absolute(path, "formal source output")
            for path in outputs.all_paths()
        )
        if len(set(paths)) != len(paths):
            raise BioasqP1FormalRuntimeError(
                "formal source output paths overlap"
            )
        root = _absolute(controller_root, "controller root")
        if not callable(getattr(hippo_lane, "start_build", None)):
            raise BioasqP1FormalRuntimeError(
                "official HippoRAG lane lacks start_build"
            )

        supplied = dict(selection_receipt)
        verify_self_hash(supplied, "selection receipt")
        persisted, receipt_raw = _read_canonical_mapping(
            outputs.safe_selection_receipt,
            mode=0o600,
            newline=True,
            field="sealed selection receipt",
        )
        if persisted != supplied:
            raise BioasqP1FormalRuntimeError(
                "sealed selection receipt differs from compiler return"
            )

        self._outputs = outputs
        self._receipt = supplied
        self._controller_root = root
        self._hippo_lane = hippo_lane
        self.selection_receipt_open_count = 1
        self.formal_marker_open_count = 0
        self.authorization_open_count = 0
        self.action_archive_open_count = {
            block: 0 for block in source.QREL_BLOCKS
        }
        self.public_open_count = {
            "corpus": 0,
            **{block: 0 for block in source.BLOCKS},
        }
        self.qrel_open_count = {
            block: 0 for block in source.QREL_BLOCKS
        }

        (
            self._public_bindings,
            self._qrel_bindings,
        ) = self._validate_selection_receipt(receipt_raw)
        self._source_commitment = stable_hash(
            {
                "p0_binding": self._receipt["p0_binding"],
                "source_access": self._receipt["source_access"],
            }
        )
        self._corpus_commitment = required_sha256(
            self._public_bindings["corpus"]["self_sha256"],
            "corpus selection commitment",
        )
        self._disjointness_commitment = stable_hash(
            {
                "disjointness_aggregate": self._receipt[
                    "disjointness_aggregate"
                ],
                "quota": self._receipt["quota"],
                "selection": self._receipt["selection"],
            }
        )
        self._qualification_commitment = required_sha256(
            self._receipt["p0_binding"][
                "public_audit_receipt_self_sha256"
            ],
            "public P0 qualification",
        )

        self._claim: ctl.AcquisitionClaim | None = None
        self._claim_attempted = False
        self._corpus: ctl.CorpusView | None = None
        self._corpus_attempted = False
        self._blocks: dict[str, ctl.BlockView] = {}
        self._block_attempted: set[str] = set()
        self._qrel_attempted: set[str] = set()
        self._m_authorization_attempted = False
        self._m_authorized = False
        self._work_ids: set[str] = set()
        self._query_groups: set[str] = set()
        self._hippo_build_started = False

    def _validate_selection_receipt(
        self,
        receipt_raw: bytes,
    ) -> tuple[
        dict[str, dict[str, object]],
        dict[str, dict[str, object]],
    ]:
        value = self._receipt
        if (
            set(value) != _SELECTION_RECEIPT_KEYS
            or value.get("schema") != source.SELECTION_RECEIPT_SCHEMA
            or value.get("status") != "selected_and_sealed"
            or value.get("study_id") != STUDY_ID
            or value.get("version") != source.VERSION
            or receipt_raw != canonical_bytes(value, newline=True)
        ):
            raise BioasqP1FormalRuntimeError(
                "selection receipt identity drifted"
            )
        artifacts = value.get("artifact_binding")
        if (
            not isinstance(artifacts, Mapping)
            or set(artifacts) != _ARTIFACT_BINDING_KEYS
            or not isinstance(artifacts.get("public_blocks"), Mapping)
            or set(artifacts["public_blocks"]) != set(source.BLOCKS)
            or not isinstance(artifacts.get("private_qrels"), Mapping)
            or set(artifacts["private_qrels"]) != set(source.QREL_BLOCKS)
        ):
            raise BioasqP1FormalRuntimeError(
                "selection receipt artifact registry drifted"
            )
        public_bindings = {
            "corpus": _validate_file_binding(
                artifacts["public_corpus"],
                expected_mode=0o600,
                expected_rows=ctl.CORPUS_SIZE,
                field="public corpus",
            )
        }
        for block in source.BLOCKS:
            public_bindings[block] = _validate_file_binding(
                artifacts["public_blocks"][block],
                expected_mode=0o400 if block == "M_search" else 0o600,
                expected_rows=ctl.BLOCK_COUNTS[block],
                field=f"{block} public block",
            )
        qrel_bindings = {
            block: _validate_file_binding(
                artifacts["private_qrels"][block],
                expected_mode=0o400,
                expected_rows=ctl.BLOCK_COUNTS[block],
                field=f"{block} private qrels",
            )
            for block in source.QREL_BLOCKS
        }
        secret = artifacts.get("private_selection_secret")
        if (
            not isinstance(secret, Mapping)
            or set(secret)
            != {
                "mode",
                "selection_secret_commitment_sha256",
                "selection_secret_persisted_publicly",
                "size_bytes",
            }
            or secret.get("mode") != "0600"
            or secret.get("size_bytes") != source.HMAC_SECRET_BYTES
            or secret.get("selection_secret_persisted_publicly") is not False
        ):
            raise BioasqP1FormalRuntimeError(
                "private selection-secret binding drifted"
            )
        required_sha256(
            secret.get("selection_secret_commitment_sha256"),
            "selection secret commitment",
        )

        if value.get("compiler_boundary") != {
            "action_count": 0,
            "model_call_count": 0,
            "online_or_API_evaluation_count": 0,
            "score_count": 0,
        }:
            raise BioasqP1FormalRuntimeError(
                "compiler action boundary drifted"
            )
        expected_quota = {
            block: {
                family: source.DEFAULT_BLOCK_FAMILY_QUOTAS[block][family]
                for family in source.FAMILIES
            }
            for block in source.BLOCKS
        }
        if value.get("quota") != expected_quota:
            raise BioasqP1FormalRuntimeError("formal quota binding drifted")
        expected_total = sum(ctl.BLOCK_COUNTS.values())
        if value.get("disjointness_aggregate") != {
            "cross_block_component_overlap_count": 0,
            "cross_block_item_overlap_count": 0,
            "cross_block_normalized_query_overlap_count": 0,
            "maximum_selected_items_per_component": 1,
            "selected_component_count": expected_total,
            "selected_item_count": expected_total,
            "selected_normalized_query_count": expected_total,
        }:
            raise BioasqP1FormalRuntimeError(
                "formal disjointness binding drifted"
            )
        selection = value.get("selection")
        if (
            not isinstance(selection, Mapping)
            or selection.get("block_order") != list(source.BLOCKS)
            or selection.get("family_order") != list(source.FAMILIES)
            or selection.get("rule") != source.SELECTION_RULE
            or selection.get("selection_secret_file_create_count") != 1
            or selection.get("selection_secret_generation_count") != 1
            or selection.get("selection_secret_persisted_publicly") is not False
            or selection.get("work_id_rule") != source.WORK_ID_RULE
        ):
            raise BioasqP1FormalRuntimeError(
                "formal selection binding drifted"
            )
        selection_commitment = required_sha256(
            selection.get("selection_secret_commitment_sha256"),
            "selection receipt secret",
        )
        if selection_commitment != secret.get(
            "selection_secret_commitment_sha256"
        ):
            raise BioasqP1FormalRuntimeError(
                "selection-secret commitments differ"
            )
        source_access = value.get("source_access")
        if source_access != {
            "file_sha256": source.OFFICIAL_SOURCE_SHA256,
            "formal_source_access_count": 1,
            "size_bytes": source.OFFICIAL_SOURCE_SIZE_BYTES,
            "source_hash_count": 1,
            "source_json_decode_count": 1,
            "source_open_count": 1,
        }:
            raise BioasqP1FormalRuntimeError(
                "formal source access binding drifted"
            )
        p0_binding = value.get("p0_binding")
        if (
            not isinstance(p0_binding, Mapping)
            or set(p0_binding)
            != {
                "implementation",
                "private_manifest_file_sha256",
                "private_manifest_self_sha256",
                "public_audit_receipt_file_sha256",
                "public_audit_receipt_self_sha256",
                "safe_receipt_file_sha256",
                "safe_receipt_self_sha256",
            }
            or p0_binding.get("private_manifest_file_sha256")
            != source.P0_PRIVATE_MANIFEST_FILE_SHA256
            or p0_binding.get("private_manifest_self_sha256")
            != source.P0_PRIVATE_MANIFEST_SELF_SHA256
            or p0_binding.get("public_audit_receipt_file_sha256")
            != source.P0_PUBLIC_AUDIT_RECEIPT_FILE_SHA256
            or p0_binding.get("public_audit_receipt_self_sha256")
            != source.P0_PUBLIC_AUDIT_RECEIPT_SELF_SHA256
            or p0_binding.get("safe_receipt_file_sha256")
            != source.P0_SAFE_RECEIPT_FILE_SHA256
            or p0_binding.get("safe_receipt_self_sha256")
            != source.P0_SAFE_RECEIPT_SELF_SHA256
            or p0_binding.get("implementation")
            != {
                "sha256": source.P0_IMPLEMENTATION_SHA256,
                "study_id": STUDY_ID,
                "version": "bioasq_p0_public_source_qualification_v1",
            }
        ):
            raise BioasqP1FormalRuntimeError("P0 binding drifted")
        if value.get("typed_core_binding") != {
            "sha256": source.TYPED_CORE_SHA256,
            "study_id": STUDY_ID,
            "version": core.VERSION,
        }:
            raise BioasqP1FormalRuntimeError("typed-core binding drifted")
        seal = value.get("seal_contract")
        if seal != {
            "M_search_open_authorization": (
                "controller_promotion_authorization_required"
            ),
            "M_search_presealed": True,
            "M_search_public_block_mode": "0400",
            "M_search_qrel_pack_mode": "0400",
            "other_late_qrel_pack_mode": "0400",
            "qrel_release_only_after_scored_block_actions_sealed": True,
        }:
            raise BioasqP1FormalRuntimeError("late-release seal drifted")
        corpus = value.get("corpus_aggregate")
        if (
            not isinstance(corpus, Mapping)
            or corpus.get("rule") != source.CORPUS_RULE
            or corpus.get("ordinal_text_row_count") != ctl.CORPUS_SIZE
            or type(corpus.get("selected_unique_qrel_count")) is not int
            or type(corpus.get("filler_unique_snippet_count")) is not int
            or corpus["selected_unique_qrel_count"] < 1
            or corpus["filler_unique_snippet_count"] < 0
            or corpus["selected_unique_qrel_count"]
            + corpus["filler_unique_snippet_count"]
            != ctl.CORPUS_SIZE
            or corpus.get("arm_corpus_file_sha256")
            != {
                arm: public_bindings["corpus"]["file_sha256"]
                for arm in ("Agent", "RAW", "official_HippoRAG")
            }
        ):
            raise BioasqP1FormalRuntimeError(
                "shared formal corpus binding drifted"
            )
        return public_bindings, qrel_bindings

    def _load_bound(
        self,
        path: Path,
        binding: Mapping[str, object],
        *,
        mode: int,
        field: str,
    ) -> dict[str, object]:
        if (
            binding.get("mode") != f"{mode:04o}"
            or type(binding.get("size_bytes")) is not int
        ):
            raise BioasqP1FormalRuntimeError(f"{field} mode binding drifted")
        value, raw = _read_canonical_mapping(
            path,
            mode=mode,
            newline=True,
            field=field,
        )
        if (
            len(raw) != binding.get("size_bytes")
            or hashlib.sha256(raw).hexdigest()
            != binding.get("file_sha256")
            or value.get("self_sha256") != binding.get("self_sha256")
        ):
            raise BioasqP1FormalRuntimeError(f"{field} binding drifted")
        return value

    def claim_formal_attempt(
        self,
        formal_marker_sha256: str,
    ) -> ctl.AcquisitionClaim:
        marker_sha = required_sha256(
            formal_marker_sha256,
            "formal marker",
        )
        if self._claim_attempted or self._claim is not None:
            raise BioasqP1FormalRuntimeError(
                "formal acquisition was claimed twice"
            )
        self._claim_attempted = True
        marker_path = self._controller_root / ctl.FORMAL_MARKER_FILENAME
        marker, _ = _read_canonical_mapping(
            marker_path,
            mode=0o400,
            newline=False,
            field="formal controller marker",
        )
        self.formal_marker_open_count += 1
        if (
            set(marker) != _MARKER_KEYS
            or marker.get("self_sha256") != marker_sha
            or marker.get("schema")
            != f"{ctl.VERSION}_one_shot_marker_v1"
            or marker.get("study_id") != STUDY_ID
            or marker.get(ctl.NO_CHANGE_COUNT_KEY) != 0
        ):
            raise BioasqP1FormalRuntimeError(
                "formal controller marker drifted"
            )
        required_sha256(
            marker.get("execution_binding_sha256"),
            "formal execution binding",
        )
        self._claim = ctl.AcquisitionClaim.create(
            source_identity_commitment=self._source_commitment,
            corpus_selection_commitment=self._corpus_commitment,
            block_disjointness_commitment=self._disjointness_commitment,
            source_qualification_commitment=(
                self._qualification_commitment
            ),
        )
        return self._claim

    def load_public_corpus(
        self,
        claim: ctl.AcquisitionClaim,
    ) -> ctl.CorpusView:
        if (
            self._claim is None
            or claim != self._claim
            or self._corpus_attempted
            or self._corpus is not None
        ):
            raise BioasqP1FormalRuntimeError(
                "public corpus lifecycle drifted"
            )
        self._corpus_attempted = True
        self.public_open_count["corpus"] += 1
        value = self._load_bound(
            self._outputs.public_corpus,
            self._public_bindings["corpus"],
            mode=0o600,
            field="public corpus",
        )
        if (
            set(value)
            != {"passages", "schema", "self_sha256", "study_id", "version"}
            or value.get("schema") != source.PUBLIC_CORPUS_SCHEMA
            or value.get("study_id") != STUDY_ID
            or value.get("version") != source.VERSION
            or not isinstance(value.get("passages"), list)
        ):
            raise BioasqP1FormalRuntimeError(
                "public corpus schema drifted"
            )
        try:
            passages = tuple(
                core.passage_from_public_fields(row)
                for row in value["passages"]
            )
            corpus = ctl.CorpusView.create(passages)
        except (
            TypeError,
            core.BioasqP1TypedCoreError,
            ctl.BioasqP1FormalControllerError,
        ) as exc:
            raise BioasqP1FormalRuntimeError(
                "public corpus typed projection drifted"
            ) from exc
        if (
            len(passages) != ctl.CORPUS_SIZE
            or self._public_bindings["corpus"].get("row_count")
            != ctl.CORPUS_SIZE
            or corpus.projection_sha256
            != core.stable_hash(value["passages"])
        ):
            raise BioasqP1FormalRuntimeError(
                "public corpus projection binding drifted"
            )
        self._corpus = corpus
        if self._hippo_build_started:
            raise BioasqP1FormalRuntimeError(
                "official HippoRAG build was started twice"
            )
        self._hippo_build_started = True
        self._hippo_lane.start_build(corpus)
        return corpus

    def _validate_m_authorization(
        self,
        authorization: Mapping[str, object] | None,
    ) -> None:
        if not isinstance(authorization, Mapping):
            raise BioasqP1FormalRuntimeError(
                "M_search requires promotion authorization"
            )
        if self._m_authorization_attempted or self._m_authorized:
            raise BioasqP1FormalRuntimeError(
                "M_search authorization lifecycle drifted"
            )
        supplied = dict(authorization)
        if set(supplied) != _AUTHORIZATION_KEYS:
            raise BioasqP1FormalRuntimeError(
                "M_search authorization schema drifted"
            )
        verify_self_hash(supplied, "M_search authorization")
        comparison_raw = supplied.get("A_hold_E1_minus_E0")
        if (
            not isinstance(comparison_raw, Mapping)
            or set(comparison_raw) != _COMPARISON_KEYS
        ):
            raise BioasqP1FormalRuntimeError(
                "M_search comparison schema drifted"
            )
        tail = comparison_raw.get(
            "one_sided_exact_magnitude_preserving_tail"
        )
        if not isinstance(tail, Mapping) or set(tail) != _FRACTION_KEYS:
            raise BioasqP1FormalRuntimeError(
                "M_search comparison tail drifted"
            )
        try:
            comparison = ctl.ExactPairedComparison(
                item_count=comparison_raw["item_count"],  # type: ignore[arg-type]
                positive_count=comparison_raw[  # type: ignore[arg-type]
                    "positive_count"
                ],
                negative_count=comparison_raw[  # type: ignore[arg-type]
                    "negative_count"
                ],
                tie_count=comparison_raw["tie_count"],  # type: ignore[arg-type]
                net_utility=comparison_raw["net_utility"],  # type: ignore[arg-type]
                one_sided_exact_magnitude_preserving_tail=Fraction(
                    tail["numerator"],  # type: ignore[arg-type]
                    tail["denominator"],  # type: ignore[arg-type]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            ctl.BioasqP1FormalControllerError,
        ) as exc:
            raise BioasqP1FormalRuntimeError(
                "M_search comparison drifted"
            ) from exc
        if (
            comparison.payload() != dict(comparison_raw)
            or comparison.item_count != ctl.BLOCK_COUNTS["A_hold"]
            or comparison.net_utility <= 0
            or comparison.one_sided_exact_magnitude_preserving_tail
            > ctl.ALPHA
            or supplied.get("schema")
            != f"{ctl.VERSION}_M_search_materialization_authorization_v1"
            or supplied.get("status") != "A_hold_E1_promoted"
            or supplied.get("study_id") != STUDY_ID
            or supplied.get("comparison_net_strictly_positive") is not True
            or supplied.get(
                "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth"
            )
            is not True
            or supplied.get("block_disjointness_commitment")
            != self._disjointness_commitment
        ):
            raise BioasqP1FormalRuntimeError(
                "M_search authorization binding drifted"
            )
        self._m_authorization_attempted = True
        authorization_path = (
            self._controller_root / ctl.PROMOTION_AUTHORIZATION_FILENAME
        )
        persisted, raw = _read_canonical_mapping(
            authorization_path,
            mode=0o400,
            newline=False,
            field="M_search promotion authorization",
        )
        self.authorization_open_count += 1
        if (
            persisted != supplied
            or raw != ctl.canonical_bytes(supplied)
        ):
            raise BioasqP1FormalRuntimeError(
                "persisted M_search authorization drifted"
            )
        self._m_authorized = True

    def load_label_free_block(
        self,
        block: str,
        authorization: Mapping[str, object] | None = None,
    ) -> ctl.BlockView:
        if self._claim is None or self._corpus is None:
            raise BioasqP1FormalRuntimeError(
                "public block requested before corpus"
            )
        if block not in source.BLOCKS or block in self._block_attempted:
            raise BioasqP1FormalRuntimeError(
                "public block lifecycle drifted"
            )
        expected = source.BLOCKS[len(self._blocks)]
        if block != expected:
            raise BioasqP1FormalRuntimeError(
                "public block load order drifted"
            )
        if block == "M_search":
            self._validate_m_authorization(authorization)
        elif authorization is not None:
            raise BioasqP1FormalRuntimeError(
                "initial public block cannot receive authorization"
            )
        self._block_attempted.add(block)
        self.public_open_count[block] += 1
        value = self._load_bound(
            self._outputs.public_blocks()[block],
            self._public_bindings[block],
            mode=0o400 if block == "M_search" else 0o600,
            field=f"{block} public block",
        )
        if (
            set(value)
            != {
                "block_id",
                "items",
                "schema",
                "self_sha256",
                "study_id",
                "version",
            }
            or value.get("schema") != source.PUBLIC_BLOCK_SCHEMA
            or value.get("study_id") != STUDY_ID
            or value.get("version") != source.VERSION
            or value.get("block_id") != block
            or not isinstance(value.get("items"), list)
        ):
            raise BioasqP1FormalRuntimeError(
                f"{block} public block schema drifted"
            )
        items: list[ctl.FormalItemView] = []
        try:
            for raw in value["items"]:
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != source.PUBLIC_ITEM_KEYS
                ):
                    raise BioasqP1FormalRuntimeError(
                        f"{block} public item schema drifted"
                    )
                items.append(
                    ctl.FormalItemView(
                        work_id=raw["work_id"],  # type: ignore[arg-type]
                        question_text=raw["query_text"],  # type: ignore[arg-type]
                    )
                )
            view = ctl.BlockView.create(block, items)
        except (
            KeyError,
            TypeError,
            core.BioasqP1TypedCoreError,
            ctl.BioasqP1FormalControllerError,
        ) as exc:
            if isinstance(exc, BioasqP1FormalRuntimeError):
                raise
            raise BioasqP1FormalRuntimeError(
                f"{block} typed item projection drifted"
            ) from exc
        if (
            len(items) != ctl.BLOCK_COUNTS[block]
            or self._public_bindings[block].get("row_count")
            != ctl.BLOCK_COUNTS[block]
        ):
            raise BioasqP1FormalRuntimeError(
                f"{block} exact public row count drifted"
            )
        for item in view.items:
            group = _query_group_key(item.question_text)
            if item.work_id in self._work_ids or group in self._query_groups:
                raise BioasqP1FormalRuntimeError(
                    "formal work/question groups are not block-disjoint"
                )
            self._work_ids.add(item.work_id)
            self._query_groups.add(group)
        self._blocks[block] = view
        if block == "M_search" and (
            len(self._work_ids) != sum(ctl.BLOCK_COUNTS.values())
            or len(self._query_groups) != sum(ctl.BLOCK_COUNTS.values())
        ):
            raise BioasqP1FormalRuntimeError(
                "complete public disjointness count drifted"
            )
        return view

    def _validate_action_archive(
        self,
        *,
        block: str,
        custody_path: Path,
        sealed_action_archive: Mapping[str, object],
    ) -> str:
        expected_path = (
            self._controller_root / f"{block}.actions.private.json"
        )
        if (
            not isinstance(custody_path, Path)
            or not custody_path.is_absolute()
            or custody_path != expected_path
            or not isinstance(sealed_action_archive, Mapping)
        ):
            raise BioasqP1FormalRuntimeError(
                "qrel release requires the exact action archive path"
            )
        archive_value = dict(sealed_action_archive)
        archive_self = verify_self_hash(
            archive_value,
            f"{block} action archive",
        )
        rows = archive_value.get("rows")
        expected_schema = (
            f"{ctl.VERSION}_A_form_private_action_archive_v1"
            if block == "A_form"
            else (
                f"{ctl.VERSION}_{block}_"
                "private_four_arm_action_archive_v1"
            )
        )
        if (
            archive_value.get("block") != block
            or archive_value.get("block_view_sha256")
            != self._blocks[block].view_sha256
            or archive_value.get("schema") != expected_schema
            or archive_value.get("study_id") != STUDY_ID
            or not isinstance(rows, list)
            or len(rows) != ctl.BLOCK_COUNTS[block]
            or any(
                not isinstance(row, Mapping)
                or not isinstance(row.get("work_id"), str)
                for row in rows
            )
        ):
            raise BioasqP1FormalRuntimeError(
                "sealed action archive semantic binding drifted"
            )
        archive_work_ids = tuple(
            row["work_id"] for row in rows  # type: ignore[index]
        )
        if (
            len(set(archive_work_ids)) != len(archive_work_ids)
            or set(archive_work_ids)
            != {item.work_id for item in self._blocks[block].items}
        ):
            raise BioasqP1FormalRuntimeError(
                "sealed action archive work registry drifted"
            )
        persisted, raw = _read_canonical_mapping(
            custody_path,
            mode=0o400,
            newline=False,
            field=f"{block} sealed action archive",
        )
        self.action_archive_open_count[block] += 1
        if (
            persisted != archive_value
            or raw != ctl.canonical_bytes(archive_value)
        ):
            raise BioasqP1FormalRuntimeError(
                "sealed action archive byte binding drifted"
            )
        return archive_self

    def release_qrels_after_action_seal(
        self,
        block: str,
        custody_path: Path,
        sealed_action_archive: Mapping[str, object],
    ) -> ctl.QrelPack:
        if (
            block not in source.QREL_BLOCKS
            or block in self._qrel_attempted
            or block not in self._blocks
            or self._corpus is None
        ):
            raise BioasqP1FormalRuntimeError(
                "late-qrel lifecycle drifted"
            )
        if block == "M_search" and not self._m_authorized:
            raise BioasqP1FormalRuntimeError(
                "M_search qrels require promotion authorization"
            )
        archive_self = self._validate_action_archive(
            block=block,
            custody_path=custody_path,
            sealed_action_archive=sealed_action_archive,
        )
        self._qrel_attempted.add(block)
        self.qrel_open_count[block] += 1
        value = self._load_bound(
            self._outputs.private_qrels()[block],
            self._qrel_bindings[block],
            mode=0o400,
            field=f"{block} private qrels",
        )
        if (
            set(value)
            != {
                "block_id",
                "qrels",
                "schema",
                "self_sha256",
                "study_id",
                "version",
            }
            or value.get("schema") != source.PRIVATE_QREL_SCHEMA
            or value.get("study_id") != STUDY_ID
            or value.get("version") != source.VERSION
            or value.get("block_id") != block
            or not isinstance(value.get("qrels"), list)
        ):
            raise BioasqP1FormalRuntimeError(
                f"{block} private qrel schema drifted"
            )
        rows: list[ctl.QrelRow] = []
        try:
            for raw in value["qrels"]:
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != source.PRIVATE_QREL_ROW_KEYS
                ):
                    raise BioasqP1FormalRuntimeError(
                        f"{block} private qrel row drifted"
                    )
                rows.append(
                    ctl.QrelRow(
                        work_id=raw["work_id"],  # type: ignore[arg-type]
                        family=raw["family"],  # type: ignore[arg-type]
                        gold_ordinals=tuple(  # type: ignore[arg-type]
                            raw["gold_ordinals"]
                        ),
                        corpus_projection_sha256=(
                            self._corpus.projection_sha256
                        ),
                    )
                )
        except (
            KeyError,
            TypeError,
            ctl.BioasqP1FormalControllerError,
        ) as exc:
            if isinstance(exc, BioasqP1FormalRuntimeError):
                raise
            raise BioasqP1FormalRuntimeError(
                f"{block} private qrel typed projection drifted"
            ) from exc
        if (
            len(rows) != ctl.BLOCK_COUNTS[block]
            or self._qrel_bindings[block].get("row_count")
            != ctl.BLOCK_COUNTS[block]
            or {row.work_id for row in rows}
            != {item.work_id for item in self._blocks[block].items}
            or Counter(row.family for row in rows)
            != Counter(source.DEFAULT_BLOCK_FAMILY_QUOTAS[block])
        ):
            raise BioasqP1FormalRuntimeError(
                f"{block} qrel coverage or family quota drifted"
            )
        return ctl.QrelPack.create(
            block=block,
            action_archive_sha256=archive_self,
            rows=rows,
        )


__all__ = ["SealedSourceAcquisitionBoundary", "VERSION"]
