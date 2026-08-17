from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction
import copy
import copyreg
import hashlib
import json
import math
from inspect import getsource, signature
import os
from pathlib import Path
import resource
import signal
import tempfile
import threading
import time
import uuid

import pytest

import hegel_machine.phase2b_formal_covert_auditor_v1 as auditor_module
from hegel_machine.hashing import stable_hash
from hegel_machine.phase2b_formal_covert_auditor_v1 import (
    AnswerabilityLabelV1,
    ChannelLeafV1,
    ChannelOnlyLeafOwnerV1,
    DecodedFormalEnvelopeV1,
    ExactProbabilityV1,
    FormalAuditAuthorityV1,
    FormalAuthorityLabelInputV1,
    FormalAuthorityLabelRowV1,
    FormalBindingV1,
    FormalCovertAuditRejectionV1,
    FormalCovertDisposition,
    FormalPrivateAlignmentLinkV1,
    FormalPrivateBatchAlignmentV1,
    FormalPrivateAuthorityBindingV1,
    FormalObservationOrderPlanV1,
    FormalRecognizerAdapterV1,
    FormalRecognizerDecisionV1,
    FormalWireAdapterV1,
    FormalWireV3AdapterV1,
    FormalUuidRenamingPlanV1,
    FormalVariantPlanV1,
    MUST_BE_ABSENT_FIELDS,
    NamespaceLeafV1,
    POLICY_BINDING_ID,
    POSITIVE_DECISION_CLAIM,
    SEMANTICS_ID,
    TRANSCRIPT_AUTHENTICATION,
    issue_development_formal_audit_authority_v1,
    issue_formal_wire_v3_audit_authority_v1,
    run_formal_covert_audit_v1,
    validate_formal_covert_audit_receipt_v1,
    validate_frozen_policy_binding_v1,
)


_HEADER = 4
_UUIDS = {
    "bundle_id": "00000000-0000-4000-8000-000000000000",
    "observation_id": "10000000-0000-4000-8000-000000000000",
    "entity_id": "20000000-0000-4000-8000-000000000000",
    "role_candidate_id": "30000000-0000-4000-8000-000000000000",
    "quantity_id": "40000000-0000-4000-8000-000000000000",
    "context_id": "50000000-0000-4000-8000-000000000000",
    "task_id": "60000000-0000-4000-8000-000000000000",
    "scale_id": "70000000-0000-4000-8000-000000000000",
    "aggregate_map_id": "80000000-0000-4000-8000-000000000000",
    "transform_id": "90000000-0000-4000-8000-000000000000",
}
_OBSERVATION_2 = "a0000000-0000-4000-8000-000000000000"
_PERMUTATION_DOMAIN_ORACLE = (
    b"HEGEL/PHASE2B/FORMAL_COVERT/TARGET_MAX_T_LABEL_PERMUTATION/V2\x00"
)
_TEST_PERMUTATION_SEED = hashlib.sha256(
    b"phase2b-formal-covert-auditor-e1-deterministic-kat"
).digest()


@pytest.fixture(autouse=True)
def _freeze_auditor_owned_permutation_seed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        auditor_module,
        "_fresh_permutation_seed",
        lambda: _TEST_PERMUTATION_SEED,
    )
    # Pytest/plugins may own native helper pthreads that are intentionally
    # outside the production fresh-process qualification contract.  A
    # dedicated negative KAT below overrides this exact inventory.
    monkeypatch.setattr(
        auditor_module,
        "_linux_process_task_ids",
        lambda: (threading.get_native_id(),),
    )
    yield
    # Receipts have an explicit bounded same-process lifecycle.  Tests release
    # every issued object rather than relying on silent eviction or GC.
    entries = tuple(auditor_module._IN_PROCESS_EXECUTED_RECEIPTS.values())
    assert all(
        type(item) is auditor_module._InProcessReceiptRegistryEntryV1
        and item.state is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        for item in entries
    )
    live = tuple(item.receipt for item in entries)
    for receipt in live:
        auditor_module.release_formal_covert_audit_receipt_v1(receipt)
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def _phase_plan_test_inputs(
    rows: int,
    partitions: tuple[object, ...],
    *,
    envelope_and_decode_bytes_bound: int | None = None,
    adapter_retained_graph_bytes_bound: int = 2 * 1_048_576,
    variant_suite_resident_bytes_per_row_bound: int | None = None,
    variant_suite_resident_fixed_bytes_bound: int = 1_048_576,
) -> tuple[object, int, object]:
    cardinalities = tuple(len(item.masks) for item in partitions)
    maximum_mask_bytes = auditor_module.sys.getsizeof((1 << rows) - 1)
    maximum_count_bytes = auditor_module.sys.getsizeof(rows)
    partition_shell_bytes = auditor_module.sys.getsizeof(
        auditor_module._Partition((), ())
    )
    projected = auditor_module.sys.getsizeof((None,) * len(cardinalities)) + sum(
        partition_shell_bytes
        + 2 * auditor_module.sys.getsizeof((None,) * cardinality)
        + cardinality * (maximum_mask_bytes + maximum_count_bytes)
        for cardinality in cardinalities
    )
    projection = auditor_module._PartitionAllocationProjectionV1(
        rows,
        cardinalities,
        projected * auditor_module.DEEP_SIZE_SAFETY_MULTIPLIER,
    )
    envelope_bound = (
        rows * auditor_module.ENVELOPE_BYTES * 2
        if envelope_and_decode_bytes_bound is None
        else envelope_and_decode_bytes_bound
    )
    per_row_bound = (
        auditor_module.ENVELOPE_BYTES
        + auditor_module.ATOMIC_VARIANT_SUITE_REFERENCE_BYTES_PER_ENVELOPE
        if variant_suite_resident_bytes_per_row_bound is None
        else variant_suite_resident_bytes_per_row_bound
    )
    invariant_plan = auditor_module._invariance_resource_plan(
        rows,
        envelope_and_decode_bytes_bound=envelope_bound,
        adapter_retained_graph_bytes_bound=adapter_retained_graph_bytes_bound,
        variant_suite_resident_bytes_per_row_bound=per_row_bound,
        variant_suite_resident_fixed_bytes_bound=(
            variant_suite_resident_fixed_bytes_bound
        ),
    )
    return (
        projection,
        time.monotonic_ns()
        + auditor_module.PARTITION_PREPARATION_TIMEOUT_SECONDS * 1_000_000_000,
        invariant_plan,
    )


def _old_two_loop_partition_oracle(
    partition: object,
    observed_masks: tuple[int, ...],
    permutation_masks: tuple[tuple[int, ...], ...],
) -> tuple[float, float, float, int]:
    cache: dict[tuple[int, ...], tuple[float, float]] = {}
    raw_nmi, advantage = auditor_module._metrics_from_masks(
        partition, observed_masks, cache
    )
    permutation_nmis = []
    for masks in permutation_masks:
        nmi, _ = auditor_module._metrics_from_masks(partition, masks, cache)
        permutation_nmis.append(nmi)
    # Independent exact-rational statement of the E1 multiset arithmetic.  The
    # oracle does not call the implementation's integer superaccumulator.
    exact_sum = sum(
        (Fraction.from_float(value) for value in (raw_nmi, *permutation_nmis)),
        Fraction(0, 1),
    )
    mean_nmi = float(exact_sum / 10_001)
    omnibus = max(
        max(0.0, raw_nmi - mean_nmi) / 0.02,
        max(0.0, advantage) / 0.05,
    )
    exceedance = 0
    for masks in permutation_masks:
        nmi, permuted_advantage = auditor_module._metrics_from_masks(
            partition, masks, cache
        )
        statistic = max(
            max(0.0, nmi - mean_nmi) / 0.02,
            max(0.0, permuted_advantage) / 0.05,
        )
        if statistic >= omnibus:
            exceedance += 1
    return raw_nmi, advantage, mean_nmi, exceedance


class _OracleSha256Stream:
    def __init__(self, seed: bytes) -> None:
        self.seed = seed
        self.counter = 0
        self.buffer = b""

    def randbelow(self, bound: int) -> int:
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            if len(self.buffer) < 8:
                self.buffer += hashlib.sha256(
                    self.seed + self.counter.to_bytes(8, "big")
                ).digest()
                self.counter += 1
            value = int.from_bytes(self.buffer[:8], "big")
            self.buffer = self.buffer[8:]
            if value < limit:
                return value % bound


def _oracle_target(row: FormalAuthorityLabelRowV1, target: str) -> tuple[str, tuple[str, ...]]:
    eligibility = "eligibility:1,2,1,1,1,2"
    if target == "family":
        return row.family, (row.case_type, row.scale, eligibility)
    if target == "binding":
        return row.binding, (row.case_type, row.family, row.scale)
    if target == "scale":
        return row.scale, (row.case_type, row.family, eligibility)
    if target == "answerable_vs_abstain":
        return row.answerability.value, (row.family, row.scale, eligibility)
    return row.joint_decision_class, (row.case_type,)


def _oracle_permutation_schedule_root(authority: FormalAuditAuthorityV1) -> str:
    hasher = hashlib.sha256(
        _PERMUTATION_DOMAIN_ORACLE
        + bytes.fromhex(SEMANTICS_ID.rsplit("_", 1)[-1])
        + hashlib.sha256(
            _PERMUTATION_DOMAIN_ORACLE + _TEST_PERMUTATION_SEED
        ).digest()
    )
    targets = (
        "family",
        "binding",
        "scale",
        "answerable_vs_abstain",
        "joint_decision_class",
    )
    for target in targets:
        pairs = tuple(_oracle_target(row, target) for row in authority.rows)
        labels = tuple(item[0] for item in pairs)
        groups: dict[tuple[str, ...], list[int]] = {}
        for index, (_, stratum) in enumerate(pairs):
            groups.setdefault(stratum, []).append(index)
        for permutation_index in range(10_000):
            for stratum in sorted(
                groups,
                key=lambda item: tuple(part.encode("utf-8") for part in item),
            ):
                positions = groups[stratum]
                donors = list(positions)
                stream = _OracleSha256Stream(
                    hashlib.sha256(
                        _PERMUTATION_DOMAIN_ORACLE
                        + _TEST_PERMUTATION_SEED
                        + target.encode("ascii")
                        + permutation_index.to_bytes(8, "big")
                        + repr(stratum).encode("utf-8")
                    ).digest()
                )
                for index in range(len(donors) - 1, 0, -1):
                    swap = stream.randbelow(index + 1)
                    donors[index], donors[swap] = donors[swap], donors[index]
                for position, donor in zip(positions, donors, strict=True):
                    hasher.update(position.to_bytes(4, "big"))
                    hasher.update(donor.to_bytes(4, "big"))
    return "phase2b_formal_covert_permutation_schedule_" + hasher.hexdigest()


def _oracle_holm(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return tuple(result)


def _oracle_loo_balanced_accuracy_advantage(
    feature: tuple[int, ...], labels: tuple[str, ...]
) -> float:
    classes = tuple(sorted(set(labels), key=lambda value: value.encode("utf-8")))
    correct = {label: 0 for label in classes}
    totals = {label: labels.count(label) for label in classes}
    for held_out, (value, true_label) in enumerate(zip(feature, labels, strict=True)):
        local = [
            labels[index]
            for index, candidate in enumerate(feature)
            if index != held_out and candidate == value
        ]
        training = local or [
            label for index, label in enumerate(labels) if index != held_out
        ]
        predicted = min(
            classes,
            key=lambda label: (-training.count(label), label.encode("utf-8")),
        )
        if predicted == true_label:
            correct[true_label] += 1
    balanced = math.fsum(correct[label] / totals[label] for label in classes) / len(classes)
    return balanced - 1.0 / len(classes)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _path_value(payload: object, path: tuple[str | int, ...]) -> object:
    value = payload
    for token in path:
        value = value[token]  # type: ignore[index]
    return value


def _channel(group: str, path: tuple[str | int, ...], payload: object) -> ChannelLeafV1:
    return ChannelLeafV1(
        group,
        path,
        stable_hash(_path_value(payload, path), prefix="phase2b_channel_leaf_"),
    )


class _SyntheticWire(FormalWireAdapterV1):
    def __init__(
        self,
        source_row_index_by_public_row: tuple[int, ...] | None = None,
        private_alignment: FormalPrivateBatchAlignmentV1 | None = None,
    ) -> None:
        self.source_row_index_by_public_row = source_row_index_by_public_row
        self._private_alignment = private_alignment

    @property
    def implementation_id(self) -> str:
        return "synthetic-wire-adapter-v1"

    @property
    def schema_id(self) -> str:
        return "synthetic-wire-schema-v1"

    def canonical_payload_bytes(self, payload: object) -> bytes:
        return _canonical(payload)

    def normalize_variant_payload(self, payload: object) -> object:
        return payload

    @property
    def private_alignment(self) -> FormalPrivateBatchAlignmentV1:
        if self._private_alignment is None:
            raise ValueError("synthetic alignment has not been batch-bound")
        return self._private_alignment

    def bind_private_authority(
        self,
        authority: FormalAuditAuthorityV1,
        envelopes: tuple[bytes, ...],
    ) -> FormalPrivateAuthorityBindingV1:
        if self._private_alignment is None:
            self._private_alignment = _alignment(
                envelopes,
                self.source_row_index_by_public_row,
                authority,
            )
        return super().bind_private_authority(authority, envelopes)

    def reencode_envelope(self, payload: object, padding_bytes: bytes) -> bytes:
        encoded = self.canonical_payload_bytes(payload)
        envelope = len(encoded).to_bytes(_HEADER, "big") + encoded + padding_bytes
        if len(envelope) != 65_536:
            raise ValueError("synthetic padding no longer closes the envelope")
        return envelope

    def decode_envelope(self, envelope: bytes) -> DecodedFormalEnvelopeV1:
        length = int.from_bytes(envelope[:_HEADER], "big")
        payload_bytes = envelope[_HEADER : _HEADER + length]
        padding = envelope[_HEADER + length :]
        payload = json.loads(payload_bytes)
        leaves = []
        for namespace in sorted(payload["namespace_ids"]):
            path = ("namespace_ids", namespace)
            leaves.append(NamespaceLeafV1(namespace, path, payload["namespace_ids"][namespace]))
        leaves.sort(key=lambda item: (item.namespace, "/" + "/".join(map(str, item.path)), item.public_uuid))
        channels = (
            _channel("control_metadata", ("control_metadata",), payload),
            _channel("descriptor_metadata", ("semantic_descriptor",), payload),
            _channel("eligibility_mask", ("eligibility_mask",), payload),
            _channel("missingness", ("missingness", "available"), payload),
            _channel("missingness", ("missingness", "value"), payload),
            _channel("provenance", ("public_provenance",), payload),
            _channel("record_order", ("observations",), payload),
            _channel("slot_order", ("observations",), payload),
        )
        owners = tuple(
            sorted(
                (
                    *(
                        ChannelOnlyLeafOwnerV1(item.path, "uuid_leaf")
                        for item in leaves
                    ),
                    ChannelOnlyLeafOwnerV1(
                        ("missingness", "available"), "missingness"
                    ),
                    ChannelOnlyLeafOwnerV1(
                        ("missingness", "value"), "missingness"
                    ),
                    ChannelOnlyLeafOwnerV1(
                        ("public_provenance",), "provenance"
                    ),
                    ChannelOnlyLeafOwnerV1(
                        ("control_metadata",), "control_metadata"
                    ),
                ),
                key=lambda item: "/" + "/".join(map(str, item.path)),
            )
        )
        return DecodedFormalEnvelopeV1(
            payload=payload,
            payload_bytes=payload_bytes,
            padding_bytes=padding,
            namespace_leaves=tuple(leaves),
            observation_array_paths=(("observations",),),
            channel_leaves=channels,
            channel_only_leaf_owners=owners,
        )


class _RecordingSyntheticWire(_SyntheticWire):
    def __init__(self) -> None:
        super().__init__()
        self.variant_plans: list[FormalVariantPlanV1] = []

    def make_variant_envelopes(
        self,
        *,
        envelopes: tuple[bytes, ...],
        decoded: tuple[DecodedFormalEnvelopeV1, ...],
        plan: FormalVariantPlanV1,
    ) -> tuple[bytes, ...]:
        self.variant_plans.append(plan)
        return super().make_variant_envelopes(
            envelopes=envelopes,
            decoded=decoded,
            plan=plan,
        )


def _payload(
    bundle_uuid: str = _UUIDS["bundle_id"],
    *,
    observation_uuid: str | None = None,
    eligibility_observation_count: int = 2,
    descriptor_segments: tuple[str, ...] = ("A fixed synthetic semantic statement.",),
) -> dict[str, object]:
    namespace_ids = {"bundle_id": bundle_uuid}
    if observation_uuid is not None:
        namespace_ids["observation_id"] = observation_uuid
    return {
        "eligibility_mask": {
            "observation_ids": [
                f"obs-{index}" for index in range(eligibility_observation_count)
            ],
            "role_candidate_ids": ["role-a"],
            "scale_ids": ["scale-a", "scale-b"],
        },
        "control_metadata": "synthetic-fixed-v1",
        "missingness": {"available": False, "value": None},
        "namespace_ids": namespace_ids,
        "observations": [
            {"observation_name": "obs-a", "value": "first"},
            {"observation_name": "obs-b", "value": "second"},
        ],
        "public_provenance": "sha256:" + "11" * 32,
        "semantic_descriptor": {
            "kind": "natural_language",
            "segments": list(descriptor_segments),
        },
        "semantic": "constant",
    }


def _envelope(payload: dict[str, object]) -> bytes:
    encoded = _canonical(payload)
    return len(encoded).to_bytes(_HEADER, "big") + encoded + bytes(65_536 - _HEADER - len(encoded))


def _envelope_with_first_padding_byte(
    payload: dict[str, object], value: int
) -> bytes:
    envelope = bytearray(_envelope(payload))
    encoded_length = int.from_bytes(envelope[:_HEADER], "big")
    envelope[_HEADER + encoded_length] = value
    return bytes(envelope)


def _alignment(
    envelopes: tuple[bytes, ...],
    source_row_index_by_public_row: tuple[int, ...] | None = None,
    authority: FormalAuditAuthorityV1 | None = None,
) -> FormalPrivateBatchAlignmentV1:
    mapping = (
        tuple(range(len(envelopes)))
        if source_row_index_by_public_row is None
        else source_row_index_by_public_row
    )
    selected_authority = _authority() if authority is None else authority
    return FormalPrivateBatchAlignmentV1(
        tuple(
            FormalPrivateAlignmentLinkV1(
                "sha256:" + hashlib.sha256(envelope).hexdigest(),
                source_index,
                selected_authority.rows[source_index].source_case_commitment,
                (),
                selected_authority.rows[source_index].label_binding_id,
            )
            for envelope, source_index in zip(envelopes, mapping, strict=True)
        ),
        None,
    )


def _source_case_commitment(index: int) -> str:
    return "sha256:" + hashlib.sha256(
        b"synthetic-formal-source-case-v1\x00" + index.to_bytes(4, "big")
    ).hexdigest()


def _authority() -> FormalAuditAuthorityV1:
    labels = []
    index = 0
    for family in ("F01", "F02"):
        for scale in ("root", "child"):
            for binding in ("binding-a", "binding-b"):
                for answerability in (
                    AnswerabilityLabelV1.ANSWERABLE,
                    AnswerabilityLabelV1.ABSTAIN,
                ):
                    for joint in ("joint-a", "joint-b"):
                        labels.append(
                            FormalAuthorityLabelInputV1(
                                index,
                                family,
                                binding,
                                scale,
                                answerability,
                                joint,
                                "main",
                            )
                        )
                        index += 1
    return issue_development_formal_audit_authority_v1(
        tuple(labels),
        tuple(_source_case_commitment(index) for index in range(len(labels))),
    )


def _insufficient_authority() -> FormalAuditAuthorityV1:
    labels = tuple(
        FormalAuthorityLabelInputV1(
            index,
            "F01",
            "binding-a",
            "root",
            AnswerabilityLabelV1.ANSWERABLE,
            "joint-a",
            "main",
        )
        for index in range(4)
    )
    return issue_development_formal_audit_authority_v1(
        labels,
        tuple(_source_case_commitment(index) for index in range(len(labels))),
    )


def _run_issuance_branch(branch: str):
    if branch not in {"sufficient", "insufficient"}:
        raise ValueError("unknown issuance branch")
    authority = _authority() if branch == "sufficient" else _insufficient_authority()
    wire = _SyntheticWire()
    return run_formal_covert_audit_v1(
        tuple(_envelope(_payload()) for _ in authority.rows),
        authority,
        wire,
        _SemanticRecognizer(wire),
    )


class _SemanticRecognizer(FormalRecognizerAdapterV1):
    def __init__(self, wire: _SyntheticWire, *, covert: bool = False) -> None:
        self.wire = wire
        self.covert = covert
        self.calls = 0

    @property
    def implementation_id(self) -> str:
        return "synthetic-semantic-recognizer-v1"

    def run_batch(self, envelopes: tuple[bytes, ...]) -> tuple[FormalRecognizerDecisionV1, ...]:
        self.calls += 1
        rows = []
        for envelope in envelopes:
            decoded = self.wire.decode_envelope(envelope)
            ids = decoded.payload["namespace_ids"]
            family = ids["bundle_id"] if self.covert else "F01"
            rows.append(
                FormalRecognizerDecisionV1(
                    "SELECT",
                    family,
                    (),
                    (),
                    b"bit-exact-residual",
                )
            )
        return tuple(rows)


class _ConstantRecognizer(FormalRecognizerAdapterV1):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def implementation_id(self) -> str:
        return "constant-rich-wire-recognizer-v1"

    def run_batch(
        self, envelopes: tuple[bytes, ...]
    ) -> tuple[FormalRecognizerDecisionV1, ...]:
        self.calls += 1
        return tuple(
            FormalRecognizerDecisionV1(
                "SELECT",
                "identity",
                (),
                (),
                b"constant-rich-wire-residual",
            )
            for _ in envelopes
        )


def _copyreg_parent_execution_sentinel(path: str) -> object:
    Path(path).write_bytes(b"PARENT_UNPICKLE_EXECUTED")
    return object()


class _ChildCopyregRecognizer(_SemanticRecognizer):
    def __init__(self, wire: _SyntheticWire, sentinel: Path) -> None:
        super().__init__(wire)
        self.sentinel = sentinel

    @property
    def implementation_id(self) -> str:
        return "child-copyreg-cow-recognizer-v1"

    def run_batch(
        self, envelopes: tuple[bytes, ...]
    ) -> tuple[FormalRecognizerDecisionV1, ...]:
        sentinel = str(self.sentinel)

        def reducer(_: object) -> tuple[object, tuple[str]]:
            return _copyreg_parent_execution_sentinel, (sentinel,)

        copyreg.pickle(auditor_module._InvariantExecutionResultV1, reducer)
        # Also mutate the child-only canonical decoder binding.  Neither
        # mutation may be visible in the parent because the fork returns only
        # bytes through a predeclared file.
        auditor_module.decode_and_require_canonical_rfc8785_v1 = (
            lambda _: _copyreg_parent_execution_sentinel(sentinel)
        )
        return super().run_batch(envelopes)


class _DriftingWire(_SyntheticWire):
    def __init__(self) -> None:
        super().__init__()
        self.implementation_id_reads = 0

    @property
    def implementation_id(self) -> str:
        self.implementation_id_reads += 1
        return (
            "drifting-wire-adapter-a-v1"
            if self.implementation_id_reads == 1
            else "drifting-wire-adapter-b-v1"
        )


class _ForgedDecisionRecognizer(FormalRecognizerAdapterV1):
    def __init__(self, decision: FormalRecognizerDecisionV1) -> None:
        self.decision = decision

    @property
    def implementation_id(self) -> str:
        return "forged-decision-recognizer-v1"

    def run_batch(
        self, envelopes: tuple[bytes, ...]
    ) -> tuple[FormalRecognizerDecisionV1, ...]:
        return tuple(self.decision for _ in envelopes)


def _forged_decision(
    *,
    binding: object = (),
    scale_set: object = (),
) -> FormalRecognizerDecisionV1:
    decision = object.__new__(FormalRecognizerDecisionV1)
    object.__setattr__(decision, "decision", "SELECT")
    object.__setattr__(decision, "family", "F01")
    object.__setattr__(decision, "binding", binding)
    object.__setattr__(decision, "scale_set", scale_set)
    object.__setattr__(decision, "candidate_residual_vector", b"bounded")
    return decision


def _valid_invariant_result(rows: int = 1) -> object:
    return auditor_module._InvariantExecutionResultV1(
        fork_input_snapshot_root=(
            "phase2b_formal_covert_invariance_input_snapshot_" + "11" * 32
        ),
        adapter_execution_pin_root=(
            "phase2b_formal_covert_adapter_execution_pins_" + "22" * 32
        ),
        invariant_execution_root=(
            "phase2b_formal_covert_invariant_execution_" + "33" * 32
        ),
        exact_invariant_comparisons=rows * 80 * 5,
        invariant_run_count=81,
        variant_suite_root=(
            "phase2b_formal_variant_suite_development_" + "44" * 32
        ),
        consumer_decode_count=rows * 80,
        variant_suite_unique_envelope_object_count=0,
        variant_suite_resident_object_graph_bytes_observed=0,
        recognizer_output_root=(
            "phase2b_formal_covert_recognizer_output_" + "55" * 32
        ),
    )


def test_public_api_has_no_caller_receipts_outputs_seed_or_thresholds():
    assert auditor_module.SCHEMA_VERSION == (
        "hegel-machine-phase2b-formal-covert-auditor/4"
    )
    profile = json.loads(
        (
            Path(__file__).parents[1]
            / "config"
            / "phase2b_formal_covert_auditor_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert profile["schema_version"] == (
        "hegel-machine-phase2b-formal-covert-auditor-profile/4"
    )
    assert profile["semantics_id"] == SEMANTICS_ID
    assert profile["policy_binding_id"] == POLICY_BINDING_ID
    assert tuple(
        auditor_module.ReceiptIssuanceRuntimeEvidenceV1.__dataclass_fields__
    ) == (
        "absolute_deadline_kind",
        "absolute_deadline_monotonic_nanoseconds",
        "absolute_deadline_timeout_seconds",
        "worker_peak_rss_bytes_sum",
        "parent_peak_rss_bytes_before_receipt_finalization",
        "parent_rss_charge_ceiling_bytes",
        "conservative_parent_plus_worker_peak_sum_bytes",
        "receipt_finalization_complete_pending_validation_count",
        "receipt_finalization_locked_deadline_check_count",
        "receipt_finalization_locked_deadline_within_absolute_limit",
        "receipt_finalization_parent_peak_within_charge_ceiling",
    )
    assert (
        auditor_module.FormalCovertAuditReceiptV1.__dataclass_fields__[
            "receipt_issuance_runtime_evidence"
        ].init
        is True
    )
    assert (
        auditor_module.FormalCovertAuditReceiptV1.__dataclass_fields__["receipt_id"].init
        is False
    )
    assert tuple(signature(run_formal_covert_audit_v1).parameters) == (
        "envelopes",
        "authority",
        "wire_adapter",
        "recognizer_adapter",
    )
    assert validate_frozen_policy_binding_v1() == POLICY_BINDING_ID


def test_public_receipt_constructor_cannot_impersonate_execution_issuance():
    forged = auditor_module.FormalCovertAuditReceiptV1()
    with pytest.raises(ValueError, match="not issued by the in-process execution factory"):
        validate_formal_covert_audit_receipt_v1(forged)


def test_receipt_registry_capacity_is_reserved_atomically_and_never_evicts():
    reservations = tuple(
        auditor_module._reserve_in_process_receipt_slot()
        for _ in range(auditor_module.MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS)
    )
    assert all(
        type(item) is auditor_module._InProcessReceiptReservationV1
        and item.entry_identity is None
        for item in reservations
    )
    candidate = object.__new__(auditor_module.FormalCovertAuditReceiptV1)
    object.__setattr__(
        candidate,
        "receipt_id",
        "phase2b_formal_covert_audit_transcript_" + "0" * 64,
    )
    auditor_module._register_preconstructed_formal_covert_audit_receipt_v1(
        _factory_token=auditor_module._RECEIPT_EXECUTION_FACTORY_TOKEN,
        _reservation=reservations[0],
        receipt=candidate,
    )
    identity = id(candidate)
    entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[identity]
    assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
    assert entry.receipt is candidate
    assert entry.receipt_id == candidate.receipt_id
    assert entry.reservation is reservations[0]
    assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
    assert reservations[0].entry_identity == identity
    assert reservations[0] not in auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS
    assert len(auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS) == len(
        reservations
    ) - 1
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_RECEIPT_REGISTRY_CAPACITY",
    ):
        auditor_module._reserve_in_process_receipt_slot()
    with pytest.raises(RuntimeError, match="identity collision"):
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1(
            _factory_token=auditor_module._RECEIPT_EXECUTION_FACTORY_TOKEN,
            _reservation=reservations[1],
            receipt=candidate,
        )
    assert auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[identity] is entry
    assert reservations[1].entry_identity is None
    assert reservations[1] in auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS
    auditor_module._rollback_in_process_receipt_issuance(reservations[0])
    assert reservations[0].entry_identity is None
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    for reservation in reservations:
        auditor_module._cancel_in_process_receipt_reservation(reservation)
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_receipt_registry_insert_failure_and_rollbacks_are_exact_and_o1(
    monkeypatch: pytest.MonkeyPatch,
):
    original_registry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS

    class NoScanRegistry(dict):
        def items(self):
            raise AssertionError("registry items scan is forbidden")

        def values(self):
            raise AssertionError("registry values scan is forbidden")

        def __iter__(self):
            raise AssertionError("registry iteration is forbidden")

    class InsertThenExitRegistry(NoScanRegistry):
        def __setitem__(self, key: object, value: object) -> None:
            super().__setitem__(key, value)
            raise SystemExit("injected SystemExit after registry insertion")

    def bare_candidate(digit: str):
        receipt = object.__new__(auditor_module.FormalCovertAuditReceiptV1)
        object.__setattr__(
            receipt,
            "receipt_id",
            "phase2b_formal_covert_audit_transcript_" + digit * 64,
        )
        return receipt

    rejecting_registry = InsertThenExitRegistry()
    monkeypatch.setattr(
        auditor_module,
        "_IN_PROCESS_EXECUTED_RECEIPTS",
        rejecting_registry,
    )
    failed_reservation = auditor_module._reserve_in_process_receipt_slot()
    failed_candidate = bare_candidate("1")
    with pytest.raises(SystemExit, match="after registry insertion"):
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1(
            _factory_token=auditor_module._RECEIPT_EXECUTION_FACTORY_TOKEN,
            _reservation=failed_reservation,
            receipt=failed_candidate,
        )
    assert not rejecting_registry
    assert failed_reservation.entry_identity is None
    assert failed_reservation in auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS
    auditor_module._cancel_in_process_receipt_reservation(failed_reservation)

    no_scan_registry = NoScanRegistry()
    monkeypatch.setattr(
        auditor_module,
        "_IN_PROCESS_EXECUTED_RECEIPTS",
        no_scan_registry,
    )
    outer_reservation = auditor_module._reserve_in_process_receipt_slot()
    outer_candidate = bare_candidate("2")
    auditor_module._register_preconstructed_formal_covert_audit_receipt_v1(
        _factory_token=auditor_module._RECEIPT_EXECUTION_FACTORY_TOKEN,
        _reservation=outer_reservation,
        receipt=outer_candidate,
    )
    auditor_module._rollback_in_process_receipt_issuance(outer_reservation)
    assert not no_scan_registry
    assert outer_reservation.entry_identity is None

    direct_reservation = auditor_module._reserve_in_process_receipt_slot()
    direct_candidate = bare_candidate("3")
    auditor_module._register_preconstructed_formal_covert_audit_receipt_v1(
        _factory_token=auditor_module._RECEIPT_EXECUTION_FACTORY_TOKEN,
        _reservation=direct_reservation,
        receipt=direct_candidate,
    )
    auditor_module._rollback_pending_in_process_receipt_issuance(
        direct_candidate,
        direct_reservation,
    )
    assert not no_scan_registry
    assert direct_reservation.entry_identity is None
    auditor_module._cancel_in_process_receipt_reservation(outer_reservation)
    auditor_module._cancel_in_process_receipt_reservation(direct_reservation)
    monkeypatch.setattr(
        auditor_module,
        "_IN_PROCESS_EXECUTED_RECEIPTS",
        original_registry,
    )

    rollback_source = getsource(auditor_module._rollback_in_process_receipt_issuance)
    direct_source = getsource(
        auditor_module._rollback_pending_in_process_receipt_issuance
    )
    for source in (rollback_source, direct_source):
        assert ".items(" not in source
        assert ".values(" not in source
        assert "tuple(" not in source


def test_receipt_finalization_charge_ceiling_accepts_rising_peaks_and_is_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline_peak = auditor_module._linux_peak_rss_bytes()
    real_statistics = auditor_module._statistics
    real_construct = auditor_module._construct_formal_covert_audit_receipt_v1
    real_register = (
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1
    )
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_statistics_deadline = (
        auditor_module._remaining_statistics_deadline_seconds
    )
    real_public_validate = auditor_module.validate_formal_covert_audit_receipt_v1
    real_release = auditor_module.release_formal_covert_audit_receipt_v1
    real_receipt_post_init = auditor_module.FormalCovertAuditReceiptV1.__post_init__
    real_feature_projection = auditor_module._target_feature_metric_projection
    real_family_projection = auditor_module._target_family_metric_projection
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    calls: Counter[str] = Counter()
    events: list[str] = []
    peak_samples: list[int] = []
    finalization_started = False
    finalization_peak_count = 0

    def tracked_statistics(*args: object, **kwargs: object):
        nonlocal finalization_started
        result = real_statistics(*args, **kwargs)
        finalization_started = True
        return result

    def increasing_peak() -> int:
        nonlocal finalization_peak_count
        value = baseline_peak + (len(peak_samples) + 1) * 4_096
        peak_samples.append(value)
        if finalization_started:
            finalization_peak_count += 1
            events.append(
                "final-pre-sample"
                if finalization_peak_count == 1
                else "final-post-sample"
            )
        return value

    def counted_construct(**values: object):
        calls["construct"] += 1
        events.append("construct")
        return real_construct(**values)

    def counted_register(**values: object):
        calls["register"] += 1
        receipt = real_register(**values)
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        events.append("pending-register")
        return receipt

    def counted_receipt_post_init(receipt: object) -> None:
        calls["receipt_post_init"] += 1
        events.append(f"receipt-post-init#{calls['receipt_post_init']}")
        return real_receipt_post_init(receipt)

    def counted_feature_projection(value: object):
        calls["feature_projection"] += 1
        return real_feature_projection(value)

    def counted_family_projection(value: object):
        calls["family_projection"] += 1
        return real_family_projection(value)

    def counted_public_validate(receipt: object):
        calls["public_validate"] += 1
        return real_public_validate(receipt)

    def counted_pending_validate(**values: object):
        calls["pending_validate"] += 1
        events.append("pending-validate")
        receipt = values["receipt"]
        with pytest.raises(ValueError, match="PENDING receipt is not publicly issued"):
            auditor_module.validate_formal_covert_audit_receipt_v1(receipt)
        calls["pending_public_rejection"] += 1
        with pytest.raises(ValueError, match="only an exact ISSUED receipt"):
            real_release(receipt)
        calls["pending_release_rejection"] += 1
        return real_pending_validate(**values)

    def tracked_statistics_deadline(deadline_ns: int) -> float:
        if auditor_module._IN_PROCESS_EXECUTED_RECEIPTS:
            entries = tuple(
                auditor_module._IN_PROCESS_EXECUTED_RECEIPTS.values()
            )
            assert len(entries) == 1
            assert (
                entries[0].state
                is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            )
            events.extend(("locked-exact-recheck", "locked-final-deadline"))
            calls["locked_deadline"] += 1
        return real_statistics_deadline(deadline_ns)

    def tracked_entry_setattr(entry: object, name: str, value: object) -> None:
        if (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        ):
            calls["commit"] += 1
            events.append("commit")
        return real_entry_setattr(entry, name, value)

    monkeypatch.setattr(auditor_module, "_statistics", tracked_statistics)
    monkeypatch.setattr(auditor_module, "_linux_peak_rss_bytes", increasing_peak)
    monkeypatch.setattr(
        auditor_module, "_construct_formal_covert_audit_receipt_v1", counted_construct
    )
    monkeypatch.setattr(
        auditor_module,
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        counted_register,
    )
    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        counted_pending_validate,
    )
    monkeypatch.setattr(
        auditor_module,
        "_remaining_statistics_deadline_seconds",
        tracked_statistics_deadline,
    )
    monkeypatch.setattr(
        auditor_module,
        "validate_formal_covert_audit_receipt_v1",
        counted_public_validate,
    )
    monkeypatch.setattr(
        auditor_module.FormalCovertAuditReceiptV1,
        "__post_init__",
        counted_receipt_post_init,
    )
    monkeypatch.setattr(
        auditor_module,
        "_target_feature_metric_projection",
        counted_feature_projection,
    )
    monkeypatch.setattr(
        auditor_module,
        "_target_family_metric_projection",
        counted_family_projection,
    )
    monkeypatch.setattr(
        auditor_module._InProcessReceiptRegistryEntryV1,
        "__setattr__",
        tracked_entry_setattr,
    )
    authority = _authority()
    wire = _SyntheticWire()
    receipt = run_formal_covert_audit_v1(
        tuple(_envelope(_payload()) for _ in authority.rows),
        authority,
        wire,
        _SemanticRecognizer(wire),
    )
    evidence = receipt.statistics_runtime_evidence
    worker_peak_sum = sum(evidence.worker_peak_rss_bytes)
    expected_ceiling = auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES - worker_peak_sum
    assert events == [
        "final-pre-sample",
        "construct",
        "receipt-post-init#1",
        "pending-register",
        "pending-validate",
        "receipt-post-init#2",
        "locked-exact-recheck",
        "locked-final-deadline",
        "final-post-sample",
        "commit",
    ]
    assert calls["construct"] == 1
    assert calls["register"] == 1
    assert calls["pending_validate"] == 1
    assert calls["locked_deadline"] == 1
    assert calls["public_validate"] == 1
    assert calls["pending_public_rejection"] == 1
    assert calls["pending_release_rejection"] == 1
    assert calls["receipt_post_init"] == 2
    assert calls["feature_projection"] == 2 * len(receipt.results)
    assert calls["family_projection"] == 2 * len(receipt.target_family_results)
    assert calls["commit"] == 1
    entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
    assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
    assert entry.receipt is receipt
    assert entry.receipt_id == receipt.receipt_id
    assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
    assert len(peak_samples) == 3
    assert peak_samples == sorted(peak_samples)
    assert evidence.parent_peak_rss_bytes_before_receipt_finalization == peak_samples[1]
    assert peak_samples[2] > peak_samples[1]
    assert peak_samples[2] <= expected_ceiling
    assert evidence.parent_rss_charge_ceiling_bytes == expected_ceiling
    assert evidence.conservative_parent_plus_worker_peak_sum_bytes == (
        expected_ceiling + worker_peak_sum
    ) == auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
    assert evidence.receipt_finalization_complete_pending_validation_count == 1
    assert evidence.receipt_finalization_parent_peak_within_charge_ceiling is True
    assert evidence.within_frozen_aggregate_cap is True
    issuance = receipt.receipt_issuance_runtime_evidence
    assert issuance.absolute_deadline_kind == "STATISTICS"
    assert issuance.absolute_deadline_timeout_seconds == (
        auditor_module.STATISTICS_WORKER_TIMEOUT_SECONDS
    )
    assert issuance.worker_peak_rss_bytes_sum == worker_peak_sum
    assert issuance.parent_peak_rss_bytes_before_receipt_finalization == (
        evidence.parent_peak_rss_bytes_before_receipt_finalization
    )
    assert issuance.parent_rss_charge_ceiling_bytes == expected_ceiling
    assert issuance.conservative_parent_plus_worker_peak_sum_bytes == (
        auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
    )
    assert issuance.receipt_finalization_complete_pending_validation_count == 1
    assert issuance.receipt_finalization_locked_deadline_check_count == 1
    assert issuance.receipt_finalization_locked_deadline_within_absolute_limit is True
    assert issuance.receipt_finalization_parent_peak_within_charge_ceiling is True

    formula_mutations = (
        (
            "parent_rss_charge_ceiling_bytes",
            expected_ceiling - 1,
            "charge ceiling formula drift",
        ),
        (
            "parent_peak_rss_bytes_before_receipt_finalization",
            expected_ceiling + 1,
            "pre-finalization parent peak exceeds",
        ),
        (
            "conservative_parent_plus_worker_peak_sum_bytes",
            auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES - 1,
            "conservative RSS charge drift",
        ),
        (
            "receipt_finalization_complete_pending_validation_count",
            2,
            "receipt finalization evidence drift",
        ),
        (
            "receipt_finalization_parent_peak_within_charge_ceiling",
            False,
            "receipt finalization evidence drift",
        ),
        (
            "worker_peak_rss_bytes",
            (auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES + 1,),
            "worker RSS sum exceeds",
        ),
    )
    for field_name, replacement, message in formula_mutations:
        original = getattr(evidence, field_name)
        object.__setattr__(evidence, field_name, replacement)
        with pytest.raises(ValueError, match=message):
            real_public_validate(receipt)
        object.__setattr__(evidence, field_name, original)

    issuance_formula_mutations = (
        (
            "parent_rss_charge_ceiling_bytes",
            expected_ceiling - 1,
            "W/C/prepeak charge formula drift",
        ),
        (
            "conservative_parent_plus_worker_peak_sum_bytes",
            auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES - 1,
            "W/C/prepeak charge formula drift",
        ),
        (
            "receipt_finalization_complete_pending_validation_count",
            2,
            "locked-finalization evidence drift",
        ),
        (
            "receipt_finalization_locked_deadline_check_count",
            0,
            "locked-finalization evidence drift",
        ),
        (
            "receipt_finalization_locked_deadline_within_absolute_limit",
            False,
            "locked-finalization evidence drift",
        ),
        (
            "receipt_finalization_parent_peak_within_charge_ceiling",
            False,
            "locked-finalization evidence drift",
        ),
    )
    for field_name, replacement, message in issuance_formula_mutations:
        original = getattr(issuance, field_name)
        object.__setattr__(issuance, field_name, replacement)
        with pytest.raises(ValueError, match=message):
            real_public_validate(receipt)
        object.__setattr__(issuance, field_name, original)
    assert real_public_validate(receipt) == receipt.receipt_id

    copied_receipt = copy.copy(receipt)
    assert copied_receipt is not receipt
    with pytest.raises(ValueError, match="not issued by the in-process execution factory"):
        real_public_validate(copied_receipt)
    real_release(receipt)


def test_public_receipt_validation_rechecks_after_concurrent_release(
    monkeypatch: pytest.MonkeyPatch,
):
    authority = _authority()
    wire = _SyntheticWire()
    receipt = run_formal_covert_audit_v1(
        tuple(_envelope(_payload()) for _ in authority.rows),
        authority,
        wire,
        _SemanticRecognizer(wire),
    )
    real_replay = auditor_module._replay_formal_covert_audit_receipt_content_v1
    replay_entered = threading.Event()
    allow_replay = threading.Event()
    outcomes: list[object] = []

    def blocked_replay(candidate: object, expected_id: str) -> str:
        replay_entered.set()
        if not allow_replay.wait(timeout=10.0):
            raise RuntimeError("concurrent validation barrier timed out")
        return real_replay(candidate, expected_id)

    def validate_in_thread() -> None:
        try:
            outcomes.append(validate_formal_covert_audit_receipt_v1(receipt))
        except BaseException as exc:
            outcomes.append(exc)

    monkeypatch.setattr(
        auditor_module,
        "_replay_formal_covert_audit_receipt_content_v1",
        blocked_replay,
    )
    validator = threading.Thread(
        target=validate_in_thread,
        name="auditor-issued-validation-race-kat",
    )
    validator.start()
    assert replay_entered.wait(timeout=10.0)
    assert auditor_module.release_formal_covert_audit_receipt_v1(receipt) == (
        receipt.receipt_id
    )
    allow_replay.set()
    validator.join(timeout=10.0)
    assert not validator.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], ValueError)
    assert "changed during full replay" in str(outcomes[0])
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_receipt_finalization_post_validation_ceiling_breach_rejects_once(
    monkeypatch: pytest.MonkeyPatch,
):
    real_statistics = auditor_module._statistics
    real_peak = auditor_module._linux_peak_rss_bytes
    real_construct = auditor_module._construct_formal_covert_audit_receipt_v1
    real_register = (
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1
    )
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    calls: Counter[str] = Counter()
    charge_ceiling: int | None = None
    finalization_peak_calls = 0

    def capture_statistics(*args: object, **kwargs: object):
        nonlocal charge_ceiling
        result = real_statistics(*args, **kwargs)
        inputs = result[-1]
        charge_ceiling = auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES - sum(
            inputs.worker_peak_rss_bytes
        )
        return result

    def staged_peak() -> int:
        nonlocal finalization_peak_calls
        if charge_ceiling is None:
            return real_peak()
        finalization_peak_calls += 1
        return charge_ceiling + (4_096 if finalization_peak_calls == 2 else 0)

    def counted_construct(**values: object):
        calls["construct"] += 1
        return real_construct(**values)

    def counted_register(**values: object):
        calls["register"] += 1
        return real_register(**values)

    def counted_pending_validate(**values: object):
        calls["pending_validate"] += 1
        return real_pending_validate(**values)

    def tracked_entry_setattr(entry: object, name: str, value: object) -> None:
        if (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        ):
            calls["commit"] += 1
        return real_entry_setattr(entry, name, value)

    monkeypatch.setattr(auditor_module, "_statistics", capture_statistics)
    monkeypatch.setattr(
        auditor_module,
        "_construct_formal_covert_audit_receipt_v1",
        counted_construct,
    )
    monkeypatch.setattr(
        auditor_module,
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        counted_register,
    )
    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        counted_pending_validate,
    )
    monkeypatch.setattr(
        auditor_module._InProcessReceiptRegistryEntryV1,
        "__setattr__",
        tracked_entry_setattr,
    )
    monkeypatch.setattr(auditor_module, "_linux_peak_rss_bytes", staged_peak)
    authority = _authority()
    wire = _SyntheticWire()
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="post-validation parent peak exceeds its RSS charge ceiling",
    ):
        run_formal_covert_audit_v1(
            tuple(_envelope(_payload()) for _ in authority.rows),
            authority,
            wire,
            _SemanticRecognizer(wire),
        )
    assert charge_ceiling is not None
    assert finalization_peak_calls == 2
    assert calls == Counter({"construct": 1, "register": 1, "pending_validate": 1})
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


@pytest.mark.parametrize(
    ("worker_peak_sum", "expected_message", "expected_parent_peak_calls"),
    (
        (
            auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES,
            "pre-finalization parent peak exceeds its RSS charge ceiling",
            2,
        ),
        (
            auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES + 1,
            "receipt finalization worker RSS sum exceeds the frozen cap",
            1,
        ),
    ),
    ids=("W-equals-cap-C-zero", "W-above-cap"),
)
def test_receipt_finalization_worker_sum_boundaries_reject_before_construction(
    monkeypatch: pytest.MonkeyPatch,
    worker_peak_sum: int,
    expected_message: str,
    expected_parent_peak_calls: int,
):
    real_statistics = auditor_module._statistics
    real_construct = auditor_module._construct_formal_covert_audit_receipt_v1
    real_register = (
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1
    )
    real_peak = auditor_module._linux_peak_rss_bytes
    calls: Counter[str] = Counter()

    def rewrite_worker_peaks(*args: object, **kwargs: object):
        result = real_statistics(*args, **kwargs)
        runtime_inputs = result[-1]
        worker_count = len(runtime_inputs.worker_peak_rss_bytes)
        rewritten_inputs = replace(
            runtime_inputs,
            worker_peak_rss_bytes=(worker_peak_sum,) + (0,) * (worker_count - 1),
        )
        return result[:-1] + (rewritten_inputs,)

    def positive_parent_peak() -> int:
        calls["parent_peak"] += 1
        value = real_peak()
        assert value > 0
        return value

    def counted_construct(**values: object):
        calls["construct"] += 1
        return real_construct(**values)

    def counted_register(**values: object):
        calls["register"] += 1
        return real_register(**values)

    monkeypatch.setattr(auditor_module, "_statistics", rewrite_worker_peaks)
    monkeypatch.setattr(auditor_module, "_linux_peak_rss_bytes", positive_parent_peak)
    monkeypatch.setattr(
        auditor_module,
        "_construct_formal_covert_audit_receipt_v1",
        counted_construct,
    )
    monkeypatch.setattr(
        auditor_module,
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        counted_register,
    )
    authority = _authority()
    wire = _SyntheticWire()
    with pytest.raises(FormalCovertAuditRejectionV1, match=expected_message):
        run_formal_covert_audit_v1(
            tuple(_envelope(_payload()) for _ in authority.rows),
            authority,
            wire,
            _SemanticRecognizer(wire),
        )
    assert calls == Counter({"parent_peak": expected_parent_peak_calls})
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


@pytest.mark.parametrize(
    "failing_hook",
    ("deadline", "peak_sample"),
)
def test_receipt_finalization_post_register_exceptions_roll_back_atomically(
    monkeypatch: pytest.MonkeyPatch,
    failing_hook: str,
):
    real_construct = auditor_module._construct_formal_covert_audit_receipt_v1
    real_register = (
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1
    )
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_deadline = auditor_module._remaining_statistics_deadline_seconds
    real_peak = auditor_module._linux_peak_rss_bytes
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    calls: Counter[str] = Counter()
    failure_message = f"injected post-register {failing_hook} failure"

    def counted_construct(**values: object):
        calls["construct"] += 1
        return real_construct(**values)

    def counted_register(**values: object):
        calls["register"] += 1
        return real_register(**values)

    def counted_pending_validate(**values: object):
        calls["pending_validate"] += 1
        return real_pending_validate(**values)

    def require_one_pending_entry() -> None:
        entries = tuple(auditor_module._IN_PROCESS_EXECUTED_RECEIPTS.values())
        assert len(entries) == 1
        assert type(entries[0]) is auditor_module._InProcessReceiptRegistryEntryV1
        assert (
            entries[0].state
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        )

    def maybe_fail_deadline(deadline_ns: int) -> float:
        if auditor_module._IN_PROCESS_EXECUTED_RECEIPTS:
            require_one_pending_entry()
            calls["deadline_failure"] += 1
            raise RuntimeError(failure_message)
        return real_deadline(deadline_ns)

    def maybe_fail_peak_sample() -> int:
        if auditor_module._IN_PROCESS_EXECUTED_RECEIPTS:
            require_one_pending_entry()
            calls["peak_sample_failure"] += 1
            raise RuntimeError(failure_message)
        return real_peak()

    def tracked_entry_setattr(entry: object, name: str, value: object) -> None:
        if (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        ):
            calls["commit"] += 1
        return real_entry_setattr(entry, name, value)

    monkeypatch.setattr(
        auditor_module,
        "_construct_formal_covert_audit_receipt_v1",
        counted_construct,
    )
    monkeypatch.setattr(
        auditor_module,
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        counted_register,
    )
    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        counted_pending_validate,
    )
    monkeypatch.setattr(
        auditor_module._InProcessReceiptRegistryEntryV1,
        "__setattr__",
        tracked_entry_setattr,
    )
    if failing_hook == "deadline":
        monkeypatch.setattr(
            auditor_module,
            "_remaining_statistics_deadline_seconds",
            maybe_fail_deadline,
        )
        failure_counter = "deadline_failure"
    else:
        monkeypatch.setattr(
            auditor_module,
            "_linux_peak_rss_bytes",
            maybe_fail_peak_sample,
        )
        failure_counter = "peak_sample_failure"
    authority = _authority()
    wire = _SyntheticWire()
    with pytest.raises(RuntimeError, match=failure_message):
        run_formal_covert_audit_v1(
            tuple(_envelope(_payload()) for _ in authority.rows),
            authority,
            wire,
            _SemanticRecognizer(wire),
        )
    assert calls == Counter(
        {
            "construct": 1,
            "register": 1,
            "pending_validate": 1,
            failure_counter: 1,
        }
    )
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_locked_final_deadline_rejects_clock_advance_while_waiting_for_registry_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_commit = auditor_module._commit_pending_formal_covert_audit_receipt_v1
    real_deadline = auditor_module._remaining_statistics_deadline_seconds
    real_monotonic_ns = time.monotonic_ns
    replay_complete = threading.Event()
    lock_acquired = threading.Event()
    commit_entered = threading.Event()
    clock_advanced = threading.Event()
    events: list[str] = []
    holder_failures: list[BaseException] = []
    holders: list[threading.Thread] = []

    def blocked_after_pending_replay(**values: object) -> str:
        validated_id = real_pending_validate(**values)
        events.append("pending-replay-complete")
        replay_complete.set()
        holder = threading.Thread(
            target=hold_registry_lock_until_commit_waits,
            name="auditor-locked-final-deadline-holder-kat",
        )
        holders.append(holder)
        holder.start()
        if not lock_acquired.wait(timeout=10.0):
            raise RuntimeError("registry lock-holder barrier timed out")
        return validated_id

    def observed_commit_entry(**values: object):
        events.append("commit-entered")
        commit_entered.set()
        return real_commit(**values)

    def controlled_monotonic_ns() -> int:
        if clock_advanced.is_set():
            return 1 << 62
        return real_monotonic_ns()

    def observed_deadline(deadline_ns: int) -> float:
        if (
            clock_advanced.is_set()
            and auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
        ):
            events.append("locked-final-deadline")
        return real_deadline(deadline_ns)

    def hold_registry_lock_until_commit_waits() -> None:
        try:
            with auditor_module._IN_PROCESS_RECEIPT_REGISTRY_LOCK:
                lock_acquired.set()
                if not commit_entered.wait(timeout=10.0):
                    raise RuntimeError("commit did not contend on the registry lock")
                events.append("clock-advanced-under-contended-lock")
                clock_advanced.set()
        except BaseException as exc:
            holder_failures.append(exc)
            lock_acquired.set()

    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        blocked_after_pending_replay,
    )
    monkeypatch.setattr(
        auditor_module,
        "_commit_pending_formal_covert_audit_receipt_v1",
        observed_commit_entry,
    )
    monkeypatch.setattr(
        auditor_module,
        "_remaining_statistics_deadline_seconds",
        observed_deadline,
    )
    monkeypatch.setattr(auditor_module.time, "monotonic_ns", controlled_monotonic_ns)
    with pytest.raises(FormalCovertAuditRejectionV1) as caught:
        _run_issuance_branch("sufficient")
    assert replay_complete.is_set()
    assert len(holders) == 1
    holder = holders[0]
    holder.join(timeout=30.0)
    assert not holder.is_alive()
    assert not holder_failures
    assert events == [
        "pending-replay-complete",
        "commit-entered",
        "clock-advanced-under-contended-lock",
        "locked-final-deadline",
    ]
    assert caught.value.code == "FAIL_STATISTICAL_RUNTIME"
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


@pytest.mark.parametrize("branch", ("sufficient", "insufficient"))
def test_shared_commit_holds_registry_lock_from_deadline_through_post_rss_and_flip(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
):
    real_statistics_deadline = (
        auditor_module._remaining_statistics_deadline_seconds
    )
    real_partition_deadline = (
        auditor_module._remaining_partition_preparation_deadline_seconds
    )
    real_peak = auditor_module._linux_peak_rss_bytes
    real_commit = auditor_module._commit_pending_formal_covert_audit_receipt_v1
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    contender_attempting = threading.Event()
    contender_acquired = threading.Event()
    calls: Counter[str] = Counter()
    events: list[str] = []
    contenders: list[threading.Thread] = []

    def contend_for_registry_lock() -> None:
        events.append("contender-attempt")
        contender_attempting.set()
        with auditor_module._IN_PROCESS_RECEIPT_REGISTRY_LOCK:
            assert calls["issued_flip_complete"] == 1
            calls["contender_acquired"] += 1
            events.append("contender-acquired-after-flip")
            contender_acquired.set()

    def observe_locked_deadline(
        deadline_ns: int,
        real_checker: object,
    ) -> float:
        if auditor_module._IN_PROCESS_EXECUTED_RECEIPTS:
            assert calls["locked_deadline"] == 0
            calls["locked_deadline"] += 1
            events.append("locked-deadline-hook")
            contender = threading.Thread(
                target=contend_for_registry_lock,
                name=f"auditor-{branch}-commit-lock-contender-kat",
            )
            contenders.append(contender)
            contender.start()
            assert contender_attempting.wait(timeout=5.0)
            assert not contender_acquired.wait(timeout=0.05)
            calls["contender_blocked_at_deadline"] += 1
            events.append("contender-blocked-at-deadline")
        return real_checker(deadline_ns)  # type: ignore[operator]

    def observed_statistics_deadline(deadline_ns: int) -> float:
        return observe_locked_deadline(deadline_ns, real_statistics_deadline)

    def observed_partition_deadline(deadline_ns: int) -> float:
        return observe_locked_deadline(deadline_ns, real_partition_deadline)

    def observed_peak() -> int:
        if calls["locked_deadline"]:
            assert contender_attempting.is_set()
            assert not contender_acquired.is_set()
            calls["post_rss"] += 1
            events.append("post-rss-with-contender-blocked")
        return real_peak()

    def observed_entry_setattr(entry: object, name: str, value: object) -> None:
        transitioning = (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        )
        if transitioning:
            assert calls["post_rss"] == 1
            assert not contender_acquired.is_set()
            calls["issued_flip"] += 1
            events.append("issued-flip-with-contender-blocked")
        real_entry_setattr(entry, name, value)
        if transitioning:
            assert not contender_acquired.is_set()
            calls["issued_flip_complete"] += 1
            events.append("issued-flip-complete-with-contender-blocked")

    def observed_shared_commit(**values: object):
        calls["shared_commit"] += 1
        receipt = real_commit(**values)
        calls["shared_commit_return"] += 1
        events.append("shared-commit-return")
        assert contender_acquired.wait(timeout=5.0)
        return receipt

    monkeypatch.setattr(
        auditor_module,
        "_remaining_statistics_deadline_seconds",
        observed_statistics_deadline,
    )
    monkeypatch.setattr(
        auditor_module,
        "_remaining_partition_preparation_deadline_seconds",
        observed_partition_deadline,
    )
    monkeypatch.setattr(auditor_module, "_linux_peak_rss_bytes", observed_peak)
    monkeypatch.setattr(
        auditor_module._InProcessReceiptRegistryEntryV1,
        "__setattr__",
        observed_entry_setattr,
    )
    monkeypatch.setattr(
        auditor_module,
        "_commit_pending_formal_covert_audit_receipt_v1",
        observed_shared_commit,
    )
    receipt = None
    try:
        receipt = _run_issuance_branch(branch)
    finally:
        for contender in contenders:
            contender.join(timeout=5.0)
            assert not contender.is_alive()
    assert receipt is not None
    assert calls == Counter(
        {
            "locked_deadline": 1,
            "contender_blocked_at_deadline": 1,
            "post_rss": 1,
            "issued_flip": 1,
            "issued_flip_complete": 1,
            "contender_acquired": 1,
            "shared_commit": 1,
            "shared_commit_return": 1,
        }
    )
    assert events.index("contender-blocked-at-deadline") < events.index(
        "post-rss-with-contender-blocked"
    )
    assert events.index("post-rss-with-contender-blocked") < events.index(
        "issued-flip-with-contender-blocked"
    )
    assert events.index("issued-flip-complete-with-contender-blocked") < events.index(
        "contender-acquired-after-flip"
    )
    assert events.index("issued-flip-complete-with-contender-blocked") < events.index(
        "shared-commit-return"
    )
    entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
    assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
    assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
    auditor_module.release_formal_covert_audit_receipt_v1(receipt)


@pytest.mark.parametrize(
    "field_name",
    (
        "parent_peak_rss_bytes_before_receipt_finalization",
        "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds",
    ),
    ids=("prepeak-plus-one", "elapsed-plus-one"),
)
def test_receipt_finalization_valid_content_mutation_rejects_by_registered_id(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
):
    real_construct = auditor_module._construct_formal_covert_audit_receipt_v1
    real_register = (
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1
    )
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    calls: Counter[str] = Counter()

    def counted_construct(**values: object):
        calls["construct"] += 1
        return real_construct(**values)

    def counted_register(**values: object):
        calls["register"] += 1
        return real_register(**values)

    def mutate_after_register(**values: object):
        calls["pending_validate"] += 1
        receipt = values["receipt"]
        assert id(receipt) in auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        evidence = receipt.statistics_runtime_evidence
        original = getattr(evidence, field_name)
        replacement = original + 1
        if field_name == "parent_peak_rss_bytes_before_receipt_finalization":
            assert replacement <= evidence.parent_rss_charge_ceiling_bytes
        else:
            assert replacement <= (
                auditor_module.STATISTICS_WORKER_TIMEOUT_SECONDS * 1_000_000_000
            )
        object.__setattr__(
            evidence,
            field_name,
            replacement,
        )
        if field_name == "parent_peak_rss_bytes_before_receipt_finalization":
            object.__setattr__(
                receipt.receipt_issuance_runtime_evidence,
                field_name,
                replacement,
            )
        calls["valid_content_mutation"] += 1
        return real_pending_validate(**values)

    def tracked_entry_setattr(entry: object, name: str, value: object) -> None:
        if (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        ):
            calls["commit"] += 1
        return real_entry_setattr(entry, name, value)

    monkeypatch.setattr(
        auditor_module, "_construct_formal_covert_audit_receipt_v1", counted_construct
    )
    monkeypatch.setattr(
        auditor_module,
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        counted_register,
    )
    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        mutate_after_register,
    )
    monkeypatch.setattr(
        auditor_module._InProcessReceiptRegistryEntryV1,
        "__setattr__",
        tracked_entry_setattr,
    )
    authority = _authority()
    wire = _SyntheticWire()
    with pytest.raises(ValueError, match="content changed after construction"):
        run_formal_covert_audit_v1(
            tuple(_envelope(_payload()) for _ in authority.rows),
            authority,
            wire,
            _SemanticRecognizer(wire),
        )
    assert calls == Counter(
        {
            "construct": 1,
            "register": 1,
            "pending_validate": 1,
            "valid_content_mutation": 1,
        }
    )
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


@pytest.mark.parametrize("branch", ("sufficient", "insufficient"))
def test_pending_commit_rejects_registered_id_drift_after_successful_replay(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
):
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    calls: Counter[str] = Counter()

    def mutate_registered_id_after_replay(**values: object) -> str:
        validated_id = real_pending_validate(**values)
        receipt = values["receipt"]
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        replacement = validated_id[:-1] + (
            "0" if validated_id[-1] != "0" else "1"
        )
        assert replacement != validated_id
        entry.receipt_id = replacement
        calls["registered_id_mutation"] += 1
        return validated_id

    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        mutate_registered_id_after_replay,
    )
    with pytest.raises(ValueError, match="locked commit identity/evidence drift"):
        _run_issuance_branch(branch)
    assert calls == Counter({"registered_id_mutation": 1})
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


@pytest.mark.parametrize("branch", ("sufficient", "insufficient"))
def test_pending_commit_rejects_candidate_id_only_drift_after_successful_replay(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
):
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    calls: Counter[str] = Counter()

    def mutate_candidate_id_only_after_replay(**values: object) -> str:
        validated_id = real_pending_validate(**values)
        receipt = values["receipt"]
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        assert entry.receipt_id == validated_id == receipt.receipt_id
        replacement = validated_id[:-1] + (
            "0" if validated_id[-1] != "0" else "1"
        )
        assert replacement.startswith("phase2b_formal_covert_audit_transcript_")
        assert replacement != validated_id
        object.__setattr__(receipt, "receipt_id", replacement)
        assert entry.receipt_id == validated_id
        calls["candidate_id_only_mutation"] += 1
        return validated_id

    def reject_any_issued_flip(entry: object, name: str, value: object) -> None:
        if (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        ):
            calls["erroneous_issued_flip"] += 1
        real_entry_setattr(entry, name, value)

    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        mutate_candidate_id_only_after_replay,
    )
    monkeypatch.setattr(
        auditor_module._InProcessReceiptRegistryEntryV1,
        "__setattr__",
        reject_any_issued_flip,
    )
    with pytest.raises(ValueError, match="locked commit identity/evidence drift"):
        _run_issuance_branch(branch)
    assert calls == Counter({"candidate_id_only_mutation": 1})
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


@pytest.mark.parametrize("branch", ("sufficient", "insufficient"))
def test_branch_neutral_issuance_evidence_valid_mutation_changes_receipt_id(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
):
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    calls: Counter[str] = Counter()

    def mutate_bound_deadline_before_replay(**values: object) -> str:
        receipt = values["receipt"]
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        evidence = receipt.receipt_issuance_runtime_evidence
        object.__setattr__(
            evidence,
            "absolute_deadline_monotonic_nanoseconds",
            evidence.absolute_deadline_monotonic_nanoseconds + 1,
        )
        calls["valid_issuance_evidence_mutation"] += 1
        return real_pending_validate(**values)

    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        mutate_bound_deadline_before_replay,
    )
    with pytest.raises(ValueError, match="content changed after construction"):
        _run_issuance_branch(branch)
    assert calls == Counter({"valid_issuance_evidence_mutation": 1})
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


@pytest.mark.parametrize("branch", ("sufficient", "insufficient"))
@pytest.mark.parametrize("failure_boundary", ("pending_replay", "issued_flip"))
def test_pending_issuance_baseexception_boundaries_roll_back_exact_entry(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    failure_boundary: str,
):
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    calls: Counter[str] = Counter()
    marker = f"injected SystemExit at {branch} {failure_boundary}"

    def exit_pending_replay(**values: object) -> str:
        receipt = values["receipt"]
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        calls["pending_replay_exit"] += 1
        raise SystemExit(marker)

    def exit_after_issued_flip(
        entry: object,
        name: str,
        value: object,
    ) -> None:
        transitioning = (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        )
        real_entry_setattr(entry, name, value)
        if transitioning:
            calls["issued_flip_exit"] += 1
            raise SystemExit(marker)

    if failure_boundary == "pending_replay":
        monkeypatch.setattr(
            auditor_module,
            "_validate_pending_formal_covert_audit_receipt_v1",
            exit_pending_replay,
        )
        expected_calls = Counter({"pending_replay_exit": 1})
    else:
        monkeypatch.setattr(
            auditor_module,
            "_validate_pending_formal_covert_audit_receipt_v1",
            real_pending_validate,
        )
        monkeypatch.setattr(
            auditor_module._InProcessReceiptRegistryEntryV1,
            "__setattr__",
            exit_after_issued_flip,
        )
        expected_calls = Counter({"issued_flip_exit": 1})
    with pytest.raises(SystemExit, match=marker):
        _run_issuance_branch(branch)
    assert calls == expected_calls
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_cleanup_faults_preserve_primary_and_outer_cancel_is_always_attempted(
    monkeypatch: pytest.MonkeyPatch,
):
    primary = KeyboardInterrupt("primary pending replay interrupt")
    calls: Counter[str] = Counter()

    def raise_primary(**values: object) -> str:
        receipt = values["receipt"]
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.PENDING
        calls["primary"] += 1
        raise primary

    def fail_direct_rollback(*args: object, **kwargs: object) -> None:
        calls["direct_rollback_failure"] += 1
        raise SystemExit("injected direct rollback SystemExit")

    def fail_outer_cancel(*args: object, **kwargs: object) -> None:
        calls["outer_cancel_failure"] += 1
        raise SystemExit("injected outer cancel SystemExit")

    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        raise_primary,
    )
    monkeypatch.setattr(
        auditor_module,
        "_rollback_pending_in_process_receipt_issuance",
        fail_direct_rollback,
    )
    monkeypatch.setattr(
        auditor_module,
        "_cancel_in_process_receipt_reservation",
        fail_outer_cancel,
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        _run_issuance_branch("insufficient")
    assert caught.value is primary
    assert calls == Counter(
        {
            "primary": 1,
            "direct_rollback_failure": 1,
            "outer_cancel_failure": 1,
        }
    )
    notes = tuple(getattr(primary, "__notes__", ()))
    assert any("direct receipt rollback cleanup failed" in note for note in notes)
    assert any("reservation cancellation cleanup failed" in note for note in notes)
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_unhanded_issued_receipt_keyboardinterrupt_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
):
    real_inner = auditor_module._run_formal_covert_audit_with_reservation_v1
    primary = KeyboardInterrupt("injected issued-before-handoff interrupt")
    unhanded: list[object] = []

    def interrupt_after_inner_issuance(*args: object, **kwargs: object):
        receipt = real_inner(*args, **kwargs)
        entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
        assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
        assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        unhanded.append(receipt)
        raise primary

    monkeypatch.setattr(
        auditor_module,
        "_run_formal_covert_audit_with_reservation_v1",
        interrupt_after_inner_issuance,
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        _run_issuance_branch("insufficient")
    assert caught.value is primary
    assert len(unhanded) == 1
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_release_deletes_registry_authority_before_backlink_interrupt(
    monkeypatch: pytest.MonkeyPatch,
):
    receipt = _run_issuance_branch("insufficient")
    identity = id(receipt)
    entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[identity]
    reservation = entry.reservation
    real_reservation_setattr = (
        auditor_module._InProcessReceiptReservationV1.__setattr__
    )
    primary = KeyboardInterrupt("injected release backlink interrupt")
    calls: Counter[str] = Counter()

    def interrupt_backlink_clear(
        slot: object,
        name: str,
        value: object,
    ) -> None:
        if (
            slot is reservation
            and name == "entry_identity"
            and value is None
            and getattr(slot, name, None) == identity
        ):
            calls["backlink_interrupt"] += 1
            raise primary
        real_reservation_setattr(slot, name, value)

    monkeypatch.setattr(
        auditor_module._InProcessReceiptReservationV1,
        "__setattr__",
        interrupt_backlink_clear,
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        auditor_module.release_formal_covert_audit_receipt_v1(receipt)
    assert caught.value is primary
    assert calls == Counter({"backlink_interrupt": 1})
    assert identity not in auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert reservation.entry_identity == identity
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS
    fresh = tuple(
        auditor_module._reserve_in_process_receipt_slot()
        for _ in range(auditor_module.MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS)
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_RECEIPT_REGISTRY_CAPACITY",
    ):
        auditor_module._reserve_in_process_receipt_slot()
    for slot in fresh:
        auditor_module._cancel_in_process_receipt_reservation(slot)
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_exact_five_target_holm_and_same_feature_e1_conjunction_kat():
    raw = tuple(
        ExactProbabilityV1(value.numerator, value.denominator)
        for value in (
            Fraction(1, 10_001),
            Fraction(2, 10_001),
            Fraction(3, 10_001),
            Fraction(1, 2),
            Fraction(1, 1),
        )
    )
    adjusted = auditor_module._holm_adjust_exact(raw)
    assert tuple(item.as_fraction() for item in adjusted) == (
        Fraction(5, 10_001),
        Fraction(8, 10_001),
        Fraction(9, 10_001),
        Fraction(1, 1),
        Fraction(1, 1),
    )
    assert auditor_module._holm_ranks_exact(raw) == (1, 2, 3, 4, 5)

    critical = ExactProbabilityV1(1, 500)

    def result(
        raw_nmi: float,
        feature_count: int,
    ) -> auditor_module.TargetFeatureResultV1:
        corrected = max(0.0, raw_nmi)
        omnibus = max(corrected / 0.02, 0.0)
        localized = Fraction(feature_count + 1, 10_001) < Fraction(1, 500)
        material = omnibus > 1.0
        witness = material and localized
        return auditor_module.TargetFeatureResultV1(
            target="family",
            feature_id="phase2b_formal_covert_feature_" + "1" * 64,
            raw_nmi=raw_nmi,
            permutation_mean_nmi=0.0,
            bias_corrected_nmi=corrected,
            balanced_accuracy_advantage=0.0,
            observed_omnibus=omnibus,
            pairwise_exceedance_count=10_000,
            pairwise_raw_p=ExactProbabilityV1(1, 1),
            pairwise_holm_adjusted_p=ExactProbabilityV1(1, 1),
            feature_max_t_exceedance_count=feature_count,
            feature_max_t_p=auditor_module._exact_probability(
                Fraction(feature_count + 1, 10_001)
            ),
            target_holm_critical_value=critical,
            target_family_holm_rejected=True,
            material_effect=material,
            feature_localization_rejected=localized,
            failure_witness=witness,
            passed=not witness,
        )

    # A material effect without localization and a localized effect exactly at
    # S=1 each PASS.  Only their conjunction on one feature is a FAIL witness.
    material_only = result(0.04, 9_999)
    equality_localized = result(0.02, 0)
    same_feature = result(0.04, 0)
    assert material_only.material_effect and not material_only.failure_witness
    assert equality_localized.observed_omnibus == 1.0
    assert equality_localized.feature_localization_rejected
    assert not equality_localized.material_effect
    assert equality_localized.passed
    assert same_feature.material_effect
    assert same_feature.feature_localization_rejected
    assert same_feature.failure_witness
    assert not same_feature.passed


def test_quota_conditioned_strata_preserve_every_permuted_signature_margin():
    authority = _authority()
    signatures = tuple(
        (1, 1, 1, 1, 1, 1)
        if row.binding == "binding-a"
        else (1, 2, 1, 1, 1, 2)
        for row in authority.rows
    )
    for target in ("family", "scale", "answerable_vs_abstain"):
        sufficiency = auditor_module._sufficiency(
            authority,
            target,
            signatures,
        )
        assert sufficiency.sufficient
        labels = tuple(
            auditor_module._target_label(row, target) for row in authority.rows
        )
        strata = tuple(
            auditor_module._target_stratum(row, target, signature)
            for row, signature in zip(
                authority.rows,
                signatures,
                strict=True,
            )
        )
        expected = Counter(zip(signatures, labels, strict=True))
        for permutation_index in range(10_000):
            permuted = auditor_module._permuted_labels(
                labels,
                strata,
                target,
                permutation_index,
                _TEST_PERMUTATION_SEED,
                None,
            )
            assert Counter(zip(signatures, permuted, strict=True)) == expected


def test_symmetric_center_is_bit_identical_when_observed_member_rotates():
    tiny = float.fromhex("0x1.0000000000000p-54")
    values = (1.0,) + (tiny,) * 10_000
    first = auditor_module._symmetric_nmi_center(values[0], values[1:])
    rotated = auditor_module._symmetric_nmi_center(
        values[-1],
        values[:-1],
    )
    middle = auditor_module._symmetric_nmi_center(
        values[5_000],
        values[:5_000] + values[5_001:],
    )
    independent = float(
        sum(
            (Fraction.from_float(value) for value in values),
            Fraction(0, 1),
        )
        / 10_001
    )
    assert first.hex() == rotated.hex() == middle.hex() == independent.hex()
    # The rejected observed-first accumulator loses every tiny term when 1.0
    # is first, proving this KAT would catch the former asymmetric arithmetic.
    asymmetric = values[0]
    for value in values[1:]:
        asymmetric += value
    assert (asymmetric / 10_001).hex() != independent.hex()


def test_known_null_passes_executes_exact_81_runs_and_proves_full_manifest():
    authority = _authority()
    wire = _RecordingSyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = tuple(_envelope(_payload()) for _ in authority.rows)
    receipt = run_formal_covert_audit_v1(envelopes, authority, wire, recognizer)
    assert receipt.disposition is FormalCovertDisposition.PASS
    assert recognizer.calls == 0
    assert receipt.invariant_run_count == 81
    assert receipt.invariant_runtime_evidence.invariant_run_count == 81
    assert receipt.invariant_runtime_evidence.process_group_empty_at_exit
    assert receipt.invariant_runtime_evidence.parent_side_adapter_sandboxed is False
    assert "COOPERATIVE_PARENT" in (
        receipt.invariant_runtime_evidence.parent_side_adapter_tcb
    )
    assert receipt.invariant_runtime_evidence.adapter_execution_pin_root == (
        receipt.adapter_execution_pin_root
    )
    assert receipt.exact_invariant_comparisons == len(envelopes) * 80 * 5
    assert receipt.invariant_execution_root.startswith(
        "phase2b_formal_covert_invariant_execution_"
    )
    # Caller-owned adapter/recognizer state is never mutated across the raw
    # fork boundary; execution coverage comes only from the closed artifact.
    assert wire.variant_plans == []
    assert receipt.holm_family_size == 5
    assert receipt.pairwise_holm_family_size == (
        5 * receipt.feature_manifest.deduplicated_feature_count
    )
    assert (
        receipt.statistics_runtime_evidence.parent_postprocess_deadline_preemptive
        is False
    )
    assert receipt.statistics_runtime_evidence.external_supervisor_required is True
    worker_peak_sum = sum(receipt.statistics_runtime_evidence.worker_peak_rss_bytes)
    assert (
        receipt.statistics_runtime_evidence.parent_peak_rss_bytes_before_receipt_finalization
        <= receipt.statistics_runtime_evidence.parent_rss_charge_ceiling_bytes
    )
    assert receipt.statistics_runtime_evidence.parent_rss_charge_ceiling_bytes == (
        auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES - worker_peak_sum
    )
    assert receipt.statistics_runtime_evidence.conservative_parent_plus_worker_peak_sum_bytes == (
        receipt.statistics_runtime_evidence.parent_rss_charge_ceiling_bytes
        + worker_peak_sum
    ) == auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
    assert (
        receipt.statistics_runtime_evidence.receipt_finalization_complete_pending_validation_count
        == 1
    )
    assert (
        receipt.statistics_runtime_evidence.receipt_finalization_parent_peak_within_charge_ceiling
        is True
    )
    issuance = receipt.receipt_issuance_runtime_evidence
    assert issuance.absolute_deadline_kind == "STATISTICS"
    assert issuance.worker_peak_rss_bytes_sum == worker_peak_sum
    assert issuance.parent_peak_rss_bytes_before_receipt_finalization == (
        receipt.statistics_runtime_evidence.parent_peak_rss_bytes_before_receipt_finalization
    )
    assert issuance.parent_rss_charge_ceiling_bytes == (
        receipt.statistics_runtime_evidence.parent_rss_charge_ceiling_bytes
    )
    assert issuance.conservative_parent_plus_worker_peak_sum_bytes == (
        auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
    )
    assert issuance.receipt_finalization_complete_pending_validation_count == 1
    assert issuance.receipt_finalization_locked_deadline_check_count == 1
    assert issuance.receipt_finalization_locked_deadline_within_absolute_limit is True
    assert issuance.receipt_finalization_parent_peak_within_charge_ceiling is True
    assert not receipt.pairwise_holm_decision_eligible
    assert not receipt.pairwise_holm_resolution_capable
    assert receipt.positive_decision_claim == POSITIVE_DECISION_CLAIM
    assert receipt.transcript_authentication == TRANSCRIPT_AUTHENTICATION
    assert tuple(item.target for item in receipt.target_family_results) == (
        "family",
        "binding",
        "scale",
        "answerable_vs_abstain",
        "joint_decision_class",
    )
    assert not any(item.holm_rejected for item in receipt.target_family_results)
    assert not any(item.failure_witness for item in receipt.results)
    assert receipt.permutation_seed_reveal_hex == _TEST_PERMUTATION_SEED.hex()
    assert receipt.permutation_seed_commitment == (
        "sha256:"
        + hashlib.sha256(
            _PERMUTATION_DOMAIN_ORACLE + _TEST_PERMUTATION_SEED
        ).hexdigest()
    )
    aliases = {
        alias
        for coverage in receipt.feature_manifest.alias_coverage
        for alias in coverage.semantic_aliases
    }
    for family in (
        "128_individual_bits",
        "first_8_16_32_bits",
        "last_8_16_32_bits",
        "hamming_weight",
        "integer_mod_3_5_7_11_13",
        "hex_character_histogram",
    ):
        assert any(family in alias for alias in aliases)
    assert receipt.absence_fields_proved == MUST_BE_ABSENT_FIELDS
    assert not receipt.authority_origin_authenticated
    assert not receipt.authority_independence_verified
    assert not receipt.actual_corpus_audit
    assert receipt.wire_issued_alignment_id is None
    assert not receipt.wire_alignment_issuance_verified
    assert receipt.eligibility_mask_quota.exact_quota_matched
    assert all(
        item.maximum_absolute_cross_product_difference == 0
        for item in receipt.eligibility_mask_quota.targets
    )
    assert receipt.permutation_schedule_root == _oracle_permutation_schedule_root(
        authority
    )
    oracle_adjusted = _oracle_holm(
        tuple(item.raw_permutation_p for item in receipt.results)
    )
    assert tuple(item.holm_adjusted_p for item in receipt.results) == pytest.approx(
        oracle_adjusted,
        abs=0.0,
    )
    record_feature = next(
        item.concrete_feature_id
        for item in receipt.feature_manifest.alias_coverage
        if any(alias.startswith("record_order/") for alias in item.semantic_aliases)
    )
    slot_feature = next(
        item.concrete_feature_id
        for item in receipt.feature_manifest.alias_coverage
        if any(alias.startswith("slot_order/") for alias in item.semantic_aliases)
    )
    assert record_feature != slot_feature
    bit_coverage = next(
        item
        for item in receipt.feature_manifest.alias_coverage
        if any("128_individual_bits/bit_000" in alias for alias in item.semantic_aliases)
    )
    assert any("first_8_16_32_bits/first_8/bit_000" in alias for alias in bit_coverage.semantic_aliases)
    with pytest.raises(TypeError, match="finite exact floats"):
        replace(receipt.results[0], raw_nmi=float("nan"))
    original_comparisons = receipt.exact_invariant_comparisons
    object.__setattr__(receipt, "exact_invariant_comparisons", original_comparisons - 1)
    with pytest.raises(ValueError, match="comparison coverage drift"):
        validate_formal_covert_audit_receipt_v1(receipt)
    object.__setattr__(receipt, "exact_invariant_comparisons", original_comparisons)
    original_invariant_root = receipt.invariant_execution_root
    object.__setattr__(receipt, "invariant_execution_root", None)
    with pytest.raises(ValueError, match="lacks the exact invariant root"):
        validate_formal_covert_audit_receipt_v1(receipt)
    object.__setattr__(receipt, "invariant_execution_root", original_invariant_root)
    original_schedule_root = receipt.permutation_schedule_root
    object.__setattr__(receipt, "permutation_schedule_root", None)
    with pytest.raises(ValueError, match="lacks the exact permutation root"):
        validate_formal_covert_audit_receipt_v1(receipt)
    object.__setattr__(receipt, "permutation_schedule_root", original_schedule_root)
    object.__setattr__(
        receipt.results[0],
        "pairwise_holm_adjusted_p",
        ExactProbabilityV1(1, 2),
    )
    with pytest.raises(ValueError, match="diagnostic pairwise Holm replay drift"):
        validate_formal_covert_audit_receipt_v1(receipt)


def test_back_to_back_audits_leave_no_worker_or_manager_thread_lifecycle():
    baseline_threads = tuple(
        (thread.ident, thread.name, thread.daemon)
        for thread in threading.enumerate()
    )
    children_path = Path(
        f"/proc/{os.getpid()}/task/{threading.get_native_id()}/children"
    )
    assert children_path.read_text(encoding="ascii").strip() == ""
    receipt_ids = []
    for _ in range(2):
        authority = _authority()
        wire = _SyntheticWire()
        receipt = run_formal_covert_audit_v1(
            tuple(_envelope(_payload()) for _ in authority.rows),
            authority,
            wire,
            _SemanticRecognizer(wire),
        )
        receipt_ids.append(receipt.receipt_id)
        auditor_module.release_formal_covert_audit_receipt_v1(receipt)
        assert children_path.read_text(encoding="ascii").strip() == ""
    assert len(receipt_ids) == 2
    assert tuple(
        (thread.ident, thread.name, thread.daemon)
        for thread in threading.enumerate()
    ) == baseline_threads


def test_known_uuid_bit_leak_returns_fail_and_matches_independent_raw_nmi_oracle():
    authority = _authority()
    source_by_public = tuple(reversed(range(len(authority.rows))))
    wire = _SyntheticWire(source_by_public)
    recognizer = _SemanticRecognizer(wire)
    first = "00000000-0000-4000-8000-000000000000"
    second = "80000000-0000-4000-8000-000000000000"
    source_envelopes = tuple(
        _envelope(_payload(second if row.family == "F02" else first))
        for row in authority.rows
    )
    envelopes = tuple(source_envelopes[index] for index in source_by_public)
    receipt = run_formal_covert_audit_v1(
        envelopes,
        authority,
        wire,
        recognizer,
    )
    assert receipt.disposition is FormalCovertDisposition.FAIL
    feature_id = next(
        item.concrete_feature_id
        for item in receipt.feature_manifest.alias_coverage
        if any(
            "uuid_leaf/bundle_id/" in alias
            and "128_individual_bits/bit_000" in alias
            for alias in item.semantic_aliases
        )
    )
    result = next(
        item
        for item in receipt.results
        if item.target == "family" and item.feature_id == feature_id
    )
    # Independent 2x2 contingency oracle: the bit and family partition are the
    # same balanced split, so MI=H(X)=H(Y)=1 and NMI=1 exactly.
    assert result.raw_nmi == pytest.approx(1.0, abs=1e-15)
    oracle_feature = tuple(1 if row.family == "F02" else 0 for row in authority.rows)
    oracle_labels = tuple(row.family for row in authority.rows)
    assert result.balanced_accuracy_advantage == pytest.approx(
        _oracle_loo_balanced_accuracy_advantage(oracle_feature, oracle_labels),
        abs=1e-15,
    )
    assert result.bias_corrected_nmi > 0.02
    assert result.material_effect
    assert result.feature_localization_rejected
    assert result.failure_witness
    target_result = next(
        item for item in receipt.target_family_results if item.target == "family"
    )
    assert target_result.holm_rejected
    assert result.feature_max_t_p.as_fraction() < (
        target_result.holm_critical_value.as_fraction()
    )
    assert not result.passed


def test_variable_uuid_shape_uses_explicit_presence_and_zero_internal_absence():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = []
    for row in authority.rows:
        parity = sum(
            (
                row.family == "F02",
                row.scale == "child",
                row.binding == "binding-b",
                row.answerability is AnswerabilityLabelV1.ABSTAIN,
                row.joint_decision_class == "joint-b",
            )
        ) % 2
        envelopes.append(
            _envelope(
                _payload(
                    observation_uuid=_UUIDS["observation_id"] if parity else None
                )
            )
        )
    envelope_tuple = tuple(envelopes)
    receipt = run_formal_covert_audit_v1(
        envelope_tuple,
        authority,
        wire,
        recognizer,
    )
    assert receipt.disposition is FormalCovertDisposition.PASS
    assert recognizer.calls == 0
    presence_id = next(
        item.concrete_feature_id
        for item in receipt.feature_manifest.alias_coverage
        if "uuid_leaf/observation_id//namespace_ids/observation_id/presence"
        in item.semantic_aliases
    )
    assert all(
        item.raw_nmi == pytest.approx(0.0, abs=1e-15)
        for item in receipt.results
        if item.feature_id == presence_id
    )


def test_uuid_path_presence_correlated_with_private_binding_returns_fail():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = tuple(
        _envelope(
            _payload(
                observation_uuid=(
                    _UUIDS["observation_id"]
                    if row.binding == "binding-b"
                    else None
                )
            )
        )
        for row in authority.rows
    )
    receipt = run_formal_covert_audit_v1(
        envelopes,
        authority,
        wire,
        recognizer,
    )
    assert receipt.disposition is FormalCovertDisposition.FAIL
    presence_id = next(
        item.concrete_feature_id
        for item in receipt.feature_manifest.alias_coverage
        if "uuid_leaf/observation_id//namespace_ids/observation_id/presence"
        in item.semantic_aliases
    )
    result = next(
        item
        for item in receipt.results
        if item.target == "binding" and item.feature_id == presence_id
    )
    assert result.raw_nmi == pytest.approx(1.0, abs=1e-15)
    assert not result.passed


def test_pairwise_xor_probe_catches_leak_with_zero_individual_bit_marginals():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = []
    for row in authority.rows:
        first = 0 if row.binding == "binding-a" else 1
        family = 0 if row.family == "F01" else 1
        second = first ^ family
        envelopes.append(
            _envelope_with_first_padding_byte(
                _payload(),
                (first << 7) | (second << 6),
            )
        )
    envelope_tuple = tuple(envelopes)
    receipt = run_formal_covert_audit_v1(
        envelope_tuple,
        authority,
        wire,
        recognizer,
    )
    assert receipt.disposition is FormalCovertDisposition.FAIL
    aliases_by_id = {
        item.concrete_feature_id: item.semantic_aliases
        for item in receipt.feature_manifest.alias_coverage
    }
    by_key = {(item.target, item.feature_id): item for item in receipt.results}
    xor_id = next(
        feature_id
        for feature_id, aliases in aliases_by_id.items()
        if "padding/prefix/byte_00/pairwise_xor_0_1" in aliases
    )
    first_bit_id = next(
        feature_id
        for feature_id, aliases in aliases_by_id.items()
        if "padding/prefix/byte_00/individual_bit_0" in aliases
    )
    second_bit_id = next(
        feature_id
        for feature_id, aliases in aliases_by_id.items()
        if "padding/prefix/byte_00/individual_bit_1" in aliases
    )
    assert by_key[("family", xor_id)].raw_nmi == pytest.approx(1.0, abs=1e-15)
    assert by_key[("family", first_bit_id)].raw_nmi == pytest.approx(0.0, abs=1e-15)
    assert by_key[("family", second_bit_id)].raw_nmi == pytest.approx(0.0, abs=1e-15)
    assert not by_key[("family", xor_id)].passed


def test_descriptor_representation_trailing_space_leak_fails_without_text_hash_feature():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = tuple(
        _envelope(
            _payload(
                descriptor_segments=(
                    "Equivalent synthetic statement."
                    + ("  " if row.binding == "binding-b" else ""),
                )
            )
        )
        for row in authority.rows
    )
    receipt = run_formal_covert_audit_v1(envelopes, authority, wire, recognizer)
    assert receipt.disposition is FormalCovertDisposition.FAIL
    descriptor_aliases = {
        alias
        for coverage in receipt.feature_manifest.alias_coverage
        for alias in coverage.semantic_aliases
        if alias.startswith("descriptor_metadata/")
    }
    required_suffixes = (
        "/kind",
        "/segment_count",
        "/representation_composite",
        "/order_position",
        "/utf8_length",
        "/trailing_ascii_spaces",
        "/first_boundary_shape",
        "/allowed_whitespace_position_codepoint_sequence",
        "/unicode_category_sequence",
        "/ascii_class_sequence",
        "/token_codepoint_length_vector",
        "/token_utf8_length_vector",
        "/nfkc_changed",
        "/token_shape_count/lower_word",
        "/token_utf8_length_bucket/5_8",
    )
    assert all(any(alias.endswith(suffix) for alias in descriptor_aliases) for suffix in required_suffixes)
    assert not any("tagged_value" in alias or "text_hash" in alias for alias in descriptor_aliases)
    trailing_id = next(
        coverage.concrete_feature_id
        for coverage in receipt.feature_manifest.alias_coverage
        if any(
            alias.endswith("/segment_position_00/trailing_ascii_spaces")
            for alias in coverage.semantic_aliases
        )
    )
    result = next(
        item
        for item in receipt.results
        if item.target == "binding" and item.feature_id == trailing_id
    )
    assert result.raw_nmi == pytest.approx(1.0, abs=1e-15)
    assert not result.passed


def test_descriptor_ucd13_validator_rejects_noncanonical_and_forbidden_surfaces():
    rejected = (
        ("", "nonempty"),
        ("a\u0301", "NFC"),
        ("a\u2000b", "NFC"),
        ("a\u2001b", "NFC"),
        ("a\u034fb", "default-ignorable"),
        ("a\ue000b", "invisible/private/undefined"),
        ("a\nf", "invisible/private/undefined"),
        ("a\ufdd0b", "RFC-8785"),
        ("row 00000000-0000-4000-8000-000000000000", "identifier/digest"),
        ("sha256:" + "a" * 64, "identifier/digest"),
        ("raw " + "b" * 64 + " token", "identifier/digest"),
    )
    for segment, message in rejected:
        with pytest.raises(ValueError, match=message):
            auditor_module._validate_descriptor_object(
                {"kind": "natural_language", "segments": [segment]}
            )

    authority = _authority()
    for non_nfc_space in ("\u2000", "\u2001"):
        wire = _SyntheticWire()
        envelopes = tuple(
            _envelope(_payload(descriptor_segments=(f"a{non_nfc_space}b",)))
            for _ in authority.rows
        )
        with pytest.raises(
            FormalCovertAuditRejectionV1,
            match="FAIL_DESCRIPTOR_METADATA_SHAPE.*NFC",
        ):
            run_formal_covert_audit_v1(
                envelopes,
                authority,
                wire,
                _SemanticRecognizer(wire),
            )


def test_descriptor_projection_tracks_exact_whitespace_position_and_token_order():
    swapped_left = auditor_module._descriptor_segment_representation(
        "a\u2002b\u2003c"
    )
    swapped_right = auditor_module._descriptor_segment_representation(
        "a\u2003b\u2002c"
    )
    assert swapped_left["allowed_whitespace_position_codepoint_sequence"] == [
        [1, 0x2002],
        [3, 0x2003],
    ]
    assert swapped_right["allowed_whitespace_position_codepoint_sequence"] == [
        [1, 0x2003],
        [3, 0x2002],
    ]
    assert (
        swapped_left["allowed_whitespace_run_signature"]
        != swapped_right["allowed_whitespace_run_signature"]
    )

    short_long = auditor_module._descriptor_segment_representation("aa bbb")
    long_short = auditor_module._descriptor_segment_representation("aaa bb")
    assert short_long["token_codepoint_length_vector"] == [2, 3]
    assert long_short["token_codepoint_length_vector"] == [3, 2]
    assert sorted(short_long["token_codepoint_length_vector"]) == sorted(
        long_short["token_codepoint_length_vector"]
    )
    assert short_long["token_utf8_length_vector"] != long_short[
        "token_utf8_length_vector"
    ]


def test_descriptor_projection_drops_lexical_and_punctuation_symbol_identity():
    assert auditor_module._descriptor_segment_representation(
        "alpha"
    ) == auditor_module._descriptor_segment_representation("omega")
    assert auditor_module._descriptor_segment_representation(
        "alpha!"
    ) == auditor_module._descriptor_segment_representation("omega?")
    assert auditor_module._descriptor_segment_representation(
        "alpha+"
    ) == auditor_module._descriptor_segment_representation("omega=")


def test_descriptor_nfc_stable_exact_whitespace_codepoint_leak_fails():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = tuple(
        _envelope(
            _payload(
                descriptor_segments=(
                    "a" + ("\u2003" if row.binding == "binding-b" else "\u2002") + "b",
                )
            )
        )
        for row in authority.rows
    )
    receipt = run_formal_covert_audit_v1(envelopes, authority, wire, recognizer)
    assert receipt.disposition is FormalCovertDisposition.FAIL
    feature_id = next(
        coverage.concrete_feature_id
        for coverage in receipt.feature_manifest.alias_coverage
        if any(
            alias.endswith(
                "/segment_position_00/allowed_whitespace_position_codepoint_sequence"
            )
            for alias in coverage.semantic_aliases
        )
    )
    result = next(
        item
        for item in receipt.results
        if item.target == "binding" and item.feature_id == feature_id
    )
    assert result.raw_nmi == pytest.approx(1.0, abs=1e-15)
    assert not result.passed


def test_descriptor_ordered_representation_composite_catches_xor_leak():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = []
    for row in authority.rows:
        family_bit = row.family == "F02"
        binding_bit = row.binding == "binding-b"
        token_order_bit = family_bit ^ binding_bit
        whitespace = "\u2003" if family_bit else "\u2002"
        text = (
            f"{whitespace}aa-bbb"
            if token_order_bit
            else f"{whitespace}aaa-bb"
        )
        envelopes.append(_envelope(_payload(descriptor_segments=(text,))))
    receipt = run_formal_covert_audit_v1(
        tuple(envelopes), authority, wire, recognizer
    )
    assert receipt.disposition is FormalCovertDisposition.FAIL
    aliases_by_id = {
        coverage.concrete_feature_id: coverage.semantic_aliases
        for coverage in receipt.feature_manifest.alias_coverage
    }
    results = {
        (item.target, item.feature_id): item for item in receipt.results
    }

    def feature_id(suffix: str) -> str:
        return next(
            concrete_id
            for concrete_id, aliases in aliases_by_id.items()
            if any(alias.endswith(suffix) for alias in aliases)
        )

    whitespace_id = feature_id(
        "/segment_position_00/allowed_whitespace_position_codepoint_sequence"
    )
    token_order_id = feature_id(
        "/segment_position_00/token_codepoint_length_vector"
    )
    composite_id = feature_id("/representation_composite")
    assert results[("binding", whitespace_id)].raw_nmi == pytest.approx(
        0.0, abs=1e-15
    )
    assert results[("binding", token_order_id)].raw_nmi == pytest.approx(
        0.0, abs=1e-15
    )
    # The composite has four representation cells and a binary target, so the
    # frozen sqrt-entropy NMI is 1/sqrt(2), while either marginal is zero.
    assert results[("binding", composite_id)].raw_nmi == pytest.approx(
        1.0 / math.sqrt(2.0), abs=1e-15
    )
    assert not results[("binding", composite_id)].passed


def test_eligibility_shape_quota_is_exact_and_rejects_family_imbalance_atomically():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    imbalanced = tuple(
        _envelope(
            _payload(
                eligibility_observation_count=(
                    1 if row.family == "F01" else 2
                )
            )
        )
        for row in authority.rows
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_ELIGIBILITY_MASK_QUOTA",
    ):
        run_formal_covert_audit_v1(
            imbalanced,
            authority,
            wire,
            recognizer,
        )
    assert recognizer.calls == 0


def test_eligibility_binding_leak_passes_quota_but_fails_global_statistics():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    envelopes = tuple(
        _envelope(
            _payload(
                eligibility_observation_count=(
                    1 if row.binding == "binding-a" else 2
                )
            )
        )
        for row in authority.rows
    )
    receipt = run_formal_covert_audit_v1(
        envelopes,
        authority,
        wire,
        recognizer,
    )
    assert receipt.eligibility_mask_quota.exact_quota_matched
    assert receipt.disposition is FormalCovertDisposition.FAIL
    eligibility_ids = {
        item.concrete_feature_id
        for item in receipt.feature_manifest.alias_coverage
        if any(
            alias.startswith("eligibility_mask/")
            for alias in item.semantic_aliases
        )
    }
    assert any(
        item.target == "binding"
        and item.feature_id in eligibility_ids
        and item.raw_nmi == pytest.approx(1.0, abs=1e-15)
        and not item.passed
        for item in receipt.results
    )


def test_insufficient_private_authority_rejects_atomically_without_recognizer_call(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline_peak = auditor_module._linux_peak_rss_bytes()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    feature_gate_calls = 0
    real_feature_gate = auditor_module._preflight_feature_extraction_memory
    real_construct = auditor_module._construct_formal_covert_audit_receipt_v1
    real_register = (
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1
    )
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    real_partition_deadline = (
        auditor_module._remaining_partition_preparation_deadline_seconds
    )
    real_receipt_post_init = auditor_module.FormalCovertAuditReceiptV1.__post_init__
    real_feature_projection = auditor_module._target_feature_metric_projection
    real_family_projection = auditor_module._target_family_metric_projection
    real_entry_setattr = auditor_module._InProcessReceiptRegistryEntryV1.__setattr__
    issuance_events: list[str] = []
    issuance_calls: Counter[str] = Counter()
    peak_samples: list[int] = []

    def observe_feature_gate(*args: object, **kwargs: object) -> object:
        nonlocal feature_gate_calls
        feature_gate_calls += 1
        return real_feature_gate(*args, **kwargs)

    def counted_construct(**values: object):
        issuance_events.append("construct")
        issuance_calls["construct"] += 1
        return real_construct(**values)

    def counted_register(**values: object):
        receipt = real_register(**values)
        issuance_events.append("pending-register")
        issuance_calls["register"] += 1
        return receipt

    def counted_pending_validate(**values: object):
        issuance_events.append("pending-validate")
        issuance_calls["pending_validate"] += 1
        return real_pending_validate(**values)

    def rising_peak() -> int:
        value = baseline_peak + (len(peak_samples) + 1) * 4_096
        peak_samples.append(value)
        issuance_events.append(
            "final-pre-sample" if len(peak_samples) == 1 else "final-post-sample"
        )
        return value

    def tracked_partition_deadline(deadline_ns: int) -> float:
        if auditor_module._IN_PROCESS_EXECUTED_RECEIPTS:
            entries = tuple(
                auditor_module._IN_PROCESS_EXECUTED_RECEIPTS.values()
            )
            assert len(entries) == 1
            assert (
                entries[0].state
                is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            )
            issuance_events.extend(
                ("locked-exact-recheck", "locked-final-deadline")
            )
            issuance_calls["locked_deadline"] += 1
        return real_partition_deadline(deadline_ns)

    def counted_receipt_post_init(receipt: object) -> None:
        issuance_calls["receipt_post_init"] += 1
        issuance_events.append(
            f"receipt-post-init#{issuance_calls['receipt_post_init']}"
        )
        return real_receipt_post_init(receipt)

    def counted_feature_projection(value: object):
        issuance_calls["feature_projection"] += 1
        return real_feature_projection(value)

    def counted_family_projection(value: object):
        issuance_calls["family_projection"] += 1
        return real_family_projection(value)

    def tracked_entry_setattr(entry: object, name: str, value: object) -> None:
        if (
            name == "state"
            and getattr(entry, "state", None)
            is auditor_module._InProcessReceiptRegistryStateV1.PENDING
            and value is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
        ):
            issuance_events.append("commit")
            issuance_calls["commit"] += 1
        return real_entry_setattr(entry, name, value)

    monkeypatch.setattr(
        auditor_module,
        "_preflight_feature_extraction_memory",
        observe_feature_gate,
    )
    monkeypatch.setattr(
        auditor_module,
        "_construct_formal_covert_audit_receipt_v1",
        counted_construct,
    )
    monkeypatch.setattr(
        auditor_module,
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        counted_register,
    )
    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        counted_pending_validate,
    )
    monkeypatch.setattr(auditor_module, "_linux_peak_rss_bytes", rising_peak)
    monkeypatch.setattr(
        auditor_module,
        "_remaining_partition_preparation_deadline_seconds",
        tracked_partition_deadline,
    )
    monkeypatch.setattr(
        auditor_module.FormalCovertAuditReceiptV1,
        "__post_init__",
        counted_receipt_post_init,
    )
    monkeypatch.setattr(
        auditor_module,
        "_target_feature_metric_projection",
        counted_feature_projection,
    )
    monkeypatch.setattr(
        auditor_module,
        "_target_family_metric_projection",
        counted_family_projection,
    )
    monkeypatch.setattr(
        auditor_module._InProcessReceiptRegistryEntryV1,
        "__setattr__",
        tracked_entry_setattr,
    )
    authority = _insufficient_authority()
    rows = authority.rows
    receipt = run_formal_covert_audit_v1(
        tuple(_envelope(_payload()) for _ in rows),
        authority,
        wire,
        recognizer,
    )
    assert receipt.disposition is FormalCovertDisposition.REJECT_INSUFFICIENT
    assert receipt.results == ()
    assert receipt.invariant_execution_root is None
    assert receipt.permutation_schedule_root is None
    assert receipt.invariant_run_count == 0
    assert recognizer.calls == 0
    assert feature_gate_calls == 2
    assert issuance_events == [
        "final-pre-sample",
        "construct",
        "receipt-post-init#1",
        "pending-register",
        "pending-validate",
        "receipt-post-init#2",
        "locked-exact-recheck",
        "locked-final-deadline",
        "final-post-sample",
        "commit",
    ]
    assert issuance_calls == Counter(
        {
            "construct": 1,
            "register": 1,
            "pending_validate": 1,
            "receipt_post_init": 2,
            "locked_deadline": 1,
            "commit": 1,
        }
    )
    issuance = receipt.receipt_issuance_runtime_evidence
    assert peak_samples == sorted(peak_samples)
    assert len(peak_samples) == 2
    assert issuance.absolute_deadline_kind == "PARTITION_PREPARATION"
    assert issuance.absolute_deadline_timeout_seconds == (
        auditor_module.PARTITION_PREPARATION_TIMEOUT_SECONDS
    )
    assert issuance.worker_peak_rss_bytes_sum == 0
    assert issuance.parent_peak_rss_bytes_before_receipt_finalization == (
        peak_samples[0]
    )
    assert issuance.parent_rss_charge_ceiling_bytes == (
        auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
    )
    assert issuance.conservative_parent_plus_worker_peak_sum_bytes == (
        auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
    )
    assert issuance.receipt_finalization_complete_pending_validation_count == 1
    assert issuance.receipt_finalization_locked_deadline_check_count == 1
    assert issuance.receipt_finalization_locked_deadline_within_absolute_limit is True
    assert issuance.receipt_finalization_parent_peak_within_charge_ceiling is True
    assert peak_samples[1] <= issuance.parent_rss_charge_ceiling_bytes
    entry = auditor_module._IN_PROCESS_EXECUTED_RECEIPTS[id(receipt)]
    assert type(entry) is auditor_module._InProcessReceiptRegistryEntryV1
    assert entry.state is auditor_module._InProcessReceiptRegistryStateV1.ISSUED
    assert validate_formal_covert_audit_receipt_v1(receipt) == receipt.receipt_id
    assert issuance_calls["receipt_post_init"] == 3
    assert issuance_calls["feature_projection"] == 0
    assert issuance_calls["family_projection"] == 0


@pytest.mark.parametrize(
    ("breach_stage", "expected_message", "expected_calls"),
    (
        (
            "pre",
            "insufficient pre-finalization parent peak exceeds its RSS charge ceiling",
            Counter({"peak": 1}),
        ),
        (
            "post",
            "post-validation parent peak exceeds its RSS charge ceiling",
            Counter(
                {
                    "peak": 2,
                    "construct": 1,
                    "register": 1,
                    "pending_validate": 1,
                }
            ),
        ),
    ),
)
def test_insufficient_issuance_pre_and_post_ceiling_boundaries_are_atomic(
    monkeypatch: pytest.MonkeyPatch,
    breach_stage: str,
    expected_message: str,
    expected_calls: Counter[str],
):
    real_construct = auditor_module._construct_formal_covert_audit_receipt_v1
    real_register = (
        auditor_module._register_preconstructed_formal_covert_audit_receipt_v1
    )
    real_pending_validate = (
        auditor_module._validate_pending_formal_covert_audit_receipt_v1
    )
    calls: Counter[str] = Counter()

    def staged_peak() -> int:
        calls["peak"] += 1
        if breach_stage == "pre" or calls["peak"] == 2:
            return auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES + 1
        return 1

    def counted_construct(**values: object):
        calls["construct"] += 1
        return real_construct(**values)

    def counted_register(**values: object):
        calls["register"] += 1
        return real_register(**values)

    def counted_pending_validate(**values: object):
        calls["pending_validate"] += 1
        return real_pending_validate(**values)

    monkeypatch.setattr(auditor_module, "_linux_peak_rss_bytes", staged_peak)
    monkeypatch.setattr(
        auditor_module,
        "_construct_formal_covert_audit_receipt_v1",
        counted_construct,
    )
    monkeypatch.setattr(
        auditor_module,
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        counted_register,
    )
    monkeypatch.setattr(
        auditor_module,
        "_validate_pending_formal_covert_audit_receipt_v1",
        counted_pending_validate,
    )
    with pytest.raises(FormalCovertAuditRejectionV1, match=expected_message):
        _run_issuance_branch("insufficient")
    assert calls == expected_calls
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_post_invariance_preparation_failure_issues_no_receipt_or_artifact(
    monkeypatch: pytest.MonkeyPatch,
):
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    events: list[str] = []
    real_feature_gate = auditor_module._preflight_feature_extraction_memory

    def observed_feature_gate(*args: object, **kwargs: object) -> object:
        events.append("feature_gate")
        return real_feature_gate(*args, **kwargs)

    def completed_invariance(*_args: object, **_kwargs: object) -> tuple[object, object]:
        # This token represents a successfully completed 81-call child phase.
        # Later preparation failure cannot retroactively claim zero callbacks.
        assert events == ["feature_gate"]
        events.append("invariance_completed")
        return object(), object()

    def fail_feature_preparation(*_args: object, **_kwargs: object) -> object:
        assert events == [
            "feature_gate",
            "invariance_completed",
            "feature_gate",
        ]
        events.append("feature_extraction")
        auditor_module._reject(
            "FAIL_INJECTED_POST_INVARIANCE_PREPARATION",
            "injected feature-preparation failure after invariant execution",
        )

    monkeypatch.setattr(auditor_module, "_run_invariance", completed_invariance)
    monkeypatch.setattr(
        auditor_module,
        "_preflight_feature_extraction_memory",
        observed_feature_gate,
    )
    monkeypatch.setattr(
        auditor_module,
        "_extract_feature_columns",
        fail_feature_preparation,
    )
    monkeypatch.setattr(
        auditor_module,
        "_acquire_private_artifact_directory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("post-invariance preparation must not acquire an artifact dir")
        ),
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_INJECTED_POST_INVARIANCE_PREPARATION",
    ):
        run_formal_covert_audit_v1(
            tuple(_envelope(_payload()) for _ in authority.rows),
            authority,
            wire,
            recognizer,
        )
    assert events == [
        "feature_gate",
        "invariance_completed",
        "feature_gate",
        "feature_extraction",
    ]
    assert not auditor_module._IN_PROCESS_EXECUTED_RECEIPTS
    assert not auditor_module._IN_PROCESS_RECEIPT_RESERVATIONS


def test_forbidden_public_field_and_caller_mutation_fail_closed():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    payload = _payload()
    payload["private_answer"] = "forbidden"
    envelopes = (_envelope(payload),) + tuple(
        _envelope(_payload()) for _ in authority.rows[1:]
    )
    with pytest.raises(FormalCovertAuditRejectionV1, match="FAIL_MUST_BE_ABSENT"):
        run_formal_covert_audit_v1(
            envelopes, authority, wire, recognizer
        )
    assert recognizer.calls == 0

    mutated = FormalAuditAuthorityV1(authority.rows)
    object.__setattr__(mutated.rows[0], "family", "MUTATED")
    with pytest.raises(FormalCovertAuditRejectionV1, match="FAIL_PRIVATE_AUTHORITY_ROOT"):
        mutated_envelopes = tuple(_envelope(_payload()) for _ in mutated.rows)
        run_formal_covert_audit_v1(
            mutated_envelopes,
            mutated,
            wire,
            recognizer,
        )


def test_adapter_cannot_mutate_the_authority_snapshot_used_downstream():
    authority = _authority()

    class MutatingWire(_SyntheticWire):
        def bind_private_authority(
            self,
            adapter_authority: FormalAuditAuthorityV1,
            envelopes: tuple[bytes, ...],
        ) -> FormalPrivateAuthorityBindingV1:
            object.__setattr__(adapter_authority.rows[0], "family", "MUTATED")
            return super().bind_private_authority(adapter_authority, envelopes)

    wire = MutatingWire(tuple(range(len(authority.rows))))
    recognizer = _SemanticRecognizer(wire)
    envelopes = tuple(_envelope(_payload()) for _ in authority.rows)
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PRIVATE_AUTHORITY_ROOT",
    ):
        run_formal_covert_audit_v1(
            envelopes,
            authority,
            wire,
            recognizer,
        )
    assert authority.rows[0].family != "MUTATED"
    assert recognizer.calls == 0


class _CovertRecognizer(_SemanticRecognizer):
    def __init__(self, wire: _SyntheticWire) -> None:
        super().__init__(wire, covert=True)


def test_a_recognizer_that_reads_uuid_bits_fails_generated_renaming_invariance():
    authority = _authority()
    wire = _SyntheticWire()
    recognizer = _CovertRecognizer(wire)
    envelopes = tuple(_envelope(_payload()) for _ in authority.rows)
    with pytest.raises(FormalCovertAuditRejectionV1, match="FAIL_INVARIANCE_RUNTIME"):
        run_formal_covert_audit_v1(
            envelopes, authority, wire, recognizer
        )
    assert recognizer.calls == 0


def test_1025_rows_fail_before_decode_or_recognizer_execution():
    wire = _SyntheticWire()
    recognizer = _SemanticRecognizer(wire)
    with pytest.raises(FormalCovertAuditRejectionV1, match="FAIL_INPUT_ROWS"):
        run_formal_covert_audit_v1(
            (_envelope(_payload()),) * 1_025,
            _authority(),
            wire,
            recognizer,
        )
    assert recognizer.calls == 0


def test_private_alignment_rejects_duplicate_source_indices_and_spliced_batch():
    authority = _authority()
    envelopes = tuple(_envelope(_payload()) for _ in authority.rows)
    valid_alignment = _alignment(envelopes, authority=authority)
    with pytest.raises(ValueError, match="exact bijection"):
        FormalPrivateBatchAlignmentV1(
            tuple(
                replace(link, source_row_index=0)
                for link in valid_alignment.links_in_public_order
            ),
            None,
        )
    alignment = valid_alignment
    spliced = list(envelopes)
    spliced[0] = spliced[0][:-1] + bytes([spliced[0][-1] ^ 1])
    recognizer = _SemanticRecognizer(_SyntheticWire())
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PRIVATE_ALIGNMENT_BATCH_BINDING",
    ):
        run_formal_covert_audit_v1(
            tuple(spliced),
            authority,
            _SyntheticWire(private_alignment=alignment),
            recognizer,
        )
    assert recognizer.calls == 0

    swapped_rows = list(authority.rows)
    swapped_rows[0] = replace(
        swapped_rows[0],
        source_case_commitment=authority.rows[1].source_case_commitment,
    )
    swapped_rows[1] = replace(
        swapped_rows[1],
        source_case_commitment=authority.rows[0].source_case_commitment,
    )
    swapped_authority = FormalAuditAuthorityV1(tuple(swapped_rows))
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PRIVATE_AUTHORITY_SOURCE_BINDING",
    ):
        run_formal_covert_audit_v1(
            envelopes,
            swapped_authority,
            _SyntheticWire(private_alignment=alignment),
            recognizer,
        )
    assert recognizer.calls == 0


def test_uuid_shaped_leaf_outside_complete_exact_10_manifest_rejects():
    authority = _authority()
    payload = _payload()
    payload["unclassified_extension_uuid"] = (
        "b0000000-0000-4000-8000-000000000000"
    )
    envelopes = (_envelope(payload),) + tuple(
        _envelope(_payload()) for _ in authority.rows[1:]
    )
    recognizer = _SemanticRecognizer(_SyntheticWire())
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_UUID_NAMESPACE_CLOSURE",
    ):
        run_formal_covert_audit_v1(
            envelopes,
            authority,
            _SyntheticWire(),
            recognizer,
        )
    assert recognizer.calls == 0


def test_wire_v3_adapter_binds_private_rows_mappings_and_build_success_schedules(
    monkeypatch: pytest.MonkeyPatch,
):
    import hegel_machine.phase2b_formal_trusted_wire_v3 as wire_v3
    import test_phase2b_formal_trusted_wire_v3 as wire_fixture

    cases = (wire_fixture.source_case(0), wire_fixture.source_case(1))
    global_catalog, catalog_binding = wire_fixture.catalog(cases)
    success = wire_v3.build_formal_trusted_wire_batch_v3(
        source_cases=cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=catalog_binding,
        run_id=b"R" * 32,
        key_sources=wire_fixture.keys(),
        key_source_receipt=wire_fixture.receipt(),
    )
    assert type(success) is wire_v3.FormalTrustedWireBuildSuccessV3
    evidence = {
        "run_id": b"R" * 32,
        "key_sources": wire_fixture.keys(),
        "source_cases": cases,
        "global_transform_catalog": global_catalog,
        "transform_catalog_binding_receipt": catalog_binding,
    }
    with pytest.raises(TypeError, match="build success"):
        FormalWireV3AdapterV1(success.public_batch, **evidence)
    with pytest.raises(ValueError, match="source commitments"):
        FormalWireV3AdapterV1(
            success,
            **{**evidence, "source_cases": tuple(reversed(cases))},
        )

    original_fresh_verify = wire_v3.verify_formal_trusted_wire_build_success_v3
    monkeypatch.setattr(
        wire_v3,
        "verify_formal_trusted_wire_build_success_v3",
        lambda **_: wire_v3.FormalTrustedWireDispositionV3.COMPLETE,
    )
    with pytest.raises(ValueError, match="exact true"):
        FormalWireV3AdapterV1(success, **evidence)
    fresh_calls: list[dict[str, object]] = []

    def record_fresh_verify(**kwargs: object) -> bool:
        fresh_calls.append(kwargs)
        return original_fresh_verify(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        wire_v3,
        "verify_formal_trusted_wire_build_success_v3",
        record_fresh_verify,
    )
    adapter = FormalWireV3AdapterV1(success, **evidence)
    assert len(fresh_calls) == 1
    assert fresh_calls[0]["build_success"] is success
    assert adapter.stop_binding_id.startswith("phase2b_formal_wire_v3_stop_binding_")
    assert adapter.fresh_build_verification_root.startswith(
        "phase2b_formal_wire_v3_fresh_build_verification_"
    )
    assert tuple(signature(FormalWireV3AdapterV1).parameters) == (
        "build_success",
        "run_id",
        "key_sources",
        "source_cases",
        "global_transform_catalog",
        "transform_catalog_binding_receipt",
    )
    labels = tuple(
        FormalAuthorityLabelInputV1(
            index,
            f"F{index}",
            f"binding-{index}",
            f"scale-{index}",
            (
                AnswerabilityLabelV1.ANSWERABLE
                if index == 0
                else AnswerabilityLabelV1.ABSTAIN
            ),
            f"joint-{index}",
            f"private-only-audit-label-{index}",
        )
        for index in range(2)
    )
    authority = issue_formal_wire_v3_audit_authority_v1(success, labels)
    binding = adapter.bind_private_authority(authority, adapter.envelopes)
    assert binding.alignment_issuance_verified
    assert binding.label_row_binding_verified
    assert success.public_batch.formal_covert_audit is False
    assert success.public_batch.key_source_transcript_verified is False
    assert success.origin_authenticated is False
    assert not hasattr(success.public_batch, "private_audit_alignment")
    assert not hasattr(success.public_batch, "variant_context")
    private_byte_tokens = tuple(
        value.encode("ascii")
        for value in (
            success.private_audit_alignment.alignment_id,
            *(
                link.source_case_commitment
                for link in success.private_audit_alignment.links_in_public_order
            ),
            *(
                link.label_binding_id
                for link in success.private_audit_alignment.links_in_public_order
            ),
            *(
                mapping.source_uuid
                for link in success.private_audit_alignment.links_in_public_order
                for mapping in link.source_to_public_uuid_rows
            ),
            *(f"private-only-audit-label-{index}" for index in range(2)),
        )
    )
    assert all(
        token not in envelope
        for envelope in adapter.envelopes
        for token in private_byte_tokens
    )
    assert all(
        row.source_case_commitment == link.source_case_commitment
        and row.label_binding_id == link.label_binding_id
        for row, link in (
            (
                authority.rows[link.source_row_index],
                link,
            )
            for link in success.private_audit_alignment.links_in_public_order
        )
    )

    schedules: list[object] = []
    original_variant = wire_v3.reencode_formal_variant_batch_v3

    def record_variant(*, build_success: object, schedule: object) -> object:
        schedules.append(schedule)
        return original_variant(
            build_success=build_success,
            schedule=schedule,
        )

    monkeypatch.setattr(
        wire_v3,
        "reencode_formal_variant_batch_v3",
        record_variant,
    )
    decoded = tuple(adapter.decode_envelope(row) for row in adapter.envelopes)
    projection = auditor_module._feature_cardinality_projection(decoded)
    manifest, _ = auditor_module._extract_feature_columns(
        adapter.envelopes,
        decoded,
        projection,
    )
    descriptor_aliases = tuple(
        alias
        for coverage in manifest.alias_coverage
        for alias in coverage.semantic_aliases
        if alias.startswith("descriptor_metadata/")
    )
    required_descriptor_suffixes = (
        "/representation_composite",
        "/allowed_whitespace_position_codepoint_sequence",
        "/allowed_whitespace_run_signature",
        "/unicode_category_sequence",
        "/ascii_class_sequence",
        "/token_codepoint_length_vector",
        "/token_utf8_length_vector",
        "/nfkc_changed",
    )
    assert all(
        any(alias.endswith(suffix) for alias in descriptor_aliases)
        for suffix in required_descriptor_suffixes
    )
    assert not any(
        "text_hash" in alias or "lexical_identity" in alias
        for alias in descriptor_aliases
    )
    maps = auditor_module._renaming_maps(
        decoded,
        auditor_module._input_root(adapter.envelopes),
        0,
    )
    renamings = tuple(
        sorted(
            FormalUuidRenamingPlanV1(namespace, source, target)
            for namespace, mapping in maps.items()
            for source, target in mapping.items()
        )
    )
    renamed = adapter.make_variant_envelopes(
        envelopes=adapter.envelopes,
        decoded=decoded,
        plan=FormalVariantPlanV1((0, 1), renamings, ()),
    )
    reordered = adapter.make_variant_envelopes(
        envelopes=adapter.envelopes,
        decoded=decoded,
        plan=FormalVariantPlanV1((1, 0), (), ()),
    )
    path = decoded[0].observation_array_paths[0]
    observation_count = len(auditor_module._resolve(decoded[0].payload, path))
    observation_variant = adapter.make_variant_envelopes(
        envelopes=adapter.envelopes,
        decoded=decoded,
        plan=FormalVariantPlanV1(
            (0, 1),
            (),
            (
                FormalObservationOrderPlanV1(
                    0,
                    path,
                    tuple(reversed(range(observation_count))),
                ),
            ),
        ),
    )
    assert all(len(rows) == 2 for rows in (renamed, reordered, observation_variant))
    assert len(schedules) == 3
    assert schedules[0].uuid_renamings
    assert schedules[1].case_order_envelope_ids == tuple(
        reversed(success.public_batch.envelope_ids)
    )
    assert schedules[2].observation_orders
    assert success.variant_context.private_alignment_id == (
        success.private_audit_alignment.alignment_id
    )

    private_alignment = adapter.private_alignment
    first_link = private_alignment.links_in_public_order[0]
    mappings = list(first_link.source_to_public_uuid_rows)
    mappings[0] = replace(mappings[0], public_uuid=mappings[1].public_uuid)
    mappings[1] = replace(mappings[1], public_uuid=first_link.source_to_public_uuid_rows[0].public_uuid)
    stale_link = replace(first_link, source_to_public_uuid_rows=tuple(mappings))
    stale_alignment = FormalPrivateBatchAlignmentV1(
        (stale_link,) + private_alignment.links_in_public_order[1:],
        private_alignment.wire_issued_alignment_id,
    )
    object.__setattr__(adapter, "_private_alignment", stale_alignment)
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PRIVATE_AUTHORITY_WIRE_BINDING",
    ):
        adapter.bind_private_authority(authority, adapter.envelopes)
    object.__setattr__(adapter, "_private_alignment", private_alignment)

    stale_rows = list(authority.rows)
    stale_rows[0] = replace(
        stale_rows[0],
        label_binding_id=authority.rows[1].label_binding_id,
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PRIVATE_AUTHORITY_SOURCE_BINDING",
    ):
        adapter.bind_private_authority(
            FormalAuditAuthorityV1(tuple(stale_rows)),
            adapter.envelopes,
        )
    spliced = (bytes([adapter.envelopes[0][0] ^ 1]) + adapter.envelopes[0][1:],) + adapter.envelopes[1:]
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PRIVATE_AUTHORITY_WIRE_BINDING",
    ):
        adapter.bind_private_authority(authority, spliced)


def test_wire_v3_bulk_amortizes_bound_state_and_matches_single_batch_path(
    monkeypatch: pytest.MonkeyPatch,
):
    import hegel_machine.phase2b_formal_trusted_wire_v3 as wire_v3
    import test_phase2b_formal_trusted_wire_v3 as wire_fixture

    cases = (wire_fixture.source_case(0), wire_fixture.source_case(1))
    global_catalog, catalog_binding = wire_fixture.catalog(cases)
    success = wire_v3.build_formal_trusted_wire_batch_v3(
        source_cases=cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=catalog_binding,
        run_id=b"B" * 32,
        key_sources=wire_fixture.keys(),
        key_source_receipt=wire_fixture.receipt(),
    )
    assert type(success) is wire_v3.FormalTrustedWireBuildSuccessV3
    adapter = FormalWireV3AdapterV1(
        success,
        run_id=b"B" * 32,
        key_sources=wire_fixture.keys(),
        source_cases=cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=catalog_binding,
    )
    decoded = tuple(adapter.decode_envelope(row) for row in adapter.envelopes)
    for item in decoded:
        auditor_module._validate_decoded(item, adapter)

    source_domain = tuple(
        sorted(
            {
                (leaf.namespace, leaf.public_uuid)
                for item in decoded
                for leaf in item.namespace_leaves
            }
        )
    )
    forbidden_uuids = {
        leaf.public_uuid
        for item in decoded
        for leaf in item.namespace_leaves
    } | {
        row.source_uuid
        for link in success.private_audit_alignment.links_in_public_order
        for row in link.source_to_public_uuid_rows
    }

    def target_uuid(trial: int, row_index: int) -> str:
        counter = 0
        while True:
            raw = hashlib.sha256(
                b"phase2b-auditor-bulk-equivalence-kat"
                + trial.to_bytes(2, "big")
                + row_index.to_bytes(2, "big")
                + counter.to_bytes(2, "big")
            ).digest()[:16]
            candidate = str(uuid.UUID(bytes=raw, version=4))
            if candidate not in forbidden_uuids:
                return candidate
            counter += 1

    plans = tuple(
        FormalVariantPlanV1(
            case_order_indices=tuple(range(len(cases))),
            uuid_renamings=tuple(
                sorted(
                    FormalUuidRenamingPlanV1(
                        namespace,
                        source_uuid,
                        target_uuid(trial, row_index),
                    )
                    for row_index, (namespace, source_uuid) in enumerate(
                        source_domain
                    )
                )
            ),
            observation_orders=(),
        )
        for trial in range(80)
    )

    build_shape_calls: list[object] = []
    plan_validation_calls: list[FormalVariantPlanV1] = []
    consumer_decode_calls: list[bytes] = []
    consumer_validate_calls: list[DecodedFormalEnvelopeV1] = []
    bulk_schedules: tuple[object, ...] | None = None
    original_build_validate = (
        wire_v3.FormalTrustedWireBuildSuccessV3._validate_issued_shape
    )
    original_decode = FormalWireV3AdapterV1.decode_envelope
    original_validate_plan = auditor_module._validate_variant_plan
    original_validate_decoded = auditor_module._validate_decoded
    original_suite = wire_v3.reencode_formal_variant_suite_v3

    def record_build_validate(build_success: object) -> None:
        if build_success is success:
            build_shape_calls.append(build_success)
        original_build_validate(build_success)  # type: ignore[arg-type]

    def record_decode(
        wire_adapter: FormalWireV3AdapterV1,
        envelope: bytes,
    ) -> DecodedFormalEnvelopeV1:
        if wire_adapter is adapter:
            consumer_decode_calls.append(envelope)
        return original_decode(wire_adapter, envelope)

    def record_validate_plan(
        plan: FormalVariantPlanV1,
        envelopes: tuple[bytes, ...],
        decoded_rows: tuple[DecodedFormalEnvelopeV1, ...],
    ) -> None:
        plan_validation_calls.append(plan)
        original_validate_plan(plan, envelopes, decoded_rows)

    def record_validate_decoded(
        item: DecodedFormalEnvelopeV1,
        wire_adapter: FormalWireAdapterV1,
    ) -> None:
        if wire_adapter is adapter:
            consumer_validate_calls.append(item)
        original_validate_decoded(item, wire_adapter)

    def record_suite(
        *, build_success: object, schedules: tuple[object, ...]
    ) -> object:
        nonlocal bulk_schedules
        bulk_schedules = schedules
        return original_suite(
            build_success=build_success,  # type: ignore[arg-type]
            schedules=schedules,  # type: ignore[arg-type]
        )

    with monkeypatch.context() as bulk_patch:
        bulk_patch.setattr(
            wire_v3.FormalTrustedWireBuildSuccessV3,
            "_validate_issued_shape",
            record_build_validate,
        )
        bulk_patch.setattr(
            FormalWireV3AdapterV1,
            "decode_envelope",
            record_decode,
        )
        bulk_patch.setattr(
            auditor_module,
            "_validate_variant_plan",
            record_validate_plan,
        )
        bulk_patch.setattr(
            auditor_module,
            "_validate_decoded",
            record_validate_decoded,
        )
        bulk_patch.setattr(
            wire_v3,
            "reencode_formal_variant_suite_v3",
            record_suite,
        )
        bulk = adapter.make_variant_suite_envelopes(
            envelopes=adapter.envelopes,
            decoded=decoded,
            plans=plans,
        )

    assert build_shape_calls == [success]
    assert tuple(plan_validation_calls) == plans
    expected_consumer_envelopes = tuple(
        envelope for batch in bulk.batches for envelope in batch
    )
    assert len(expected_consumer_envelopes) == len(cases) * 80
    assert tuple(consumer_decode_calls) == expected_consumer_envelopes
    assert len(consumer_validate_calls) == len(expected_consumer_envelopes)
    assert all(
        envelope
        == envelope[: wire_v3.ENVELOPE_HEADER_BYTES_V3]
        + decoded_row.payload_bytes
        + decoded_row.padding_bytes
        for envelope, decoded_row in zip(
            expected_consumer_envelopes,
            consumer_validate_calls,
            strict=True,
        )
    )
    assert bulk.consumer_decode_count == len(cases) * 80
    assert bulk_schedules is not None

    single_schedules: list[object] = []
    original_single = wire_v3.reencode_formal_variant_batch_v3

    def record_single(
        *, build_success: object, schedule: object
    ) -> tuple[bytes, ...]:
        single_schedules.append(schedule)
        return original_single(
            build_success=build_success,  # type: ignore[arg-type]
            schedule=schedule,  # type: ignore[arg-type]
        )

    with monkeypatch.context() as single_patch:
        single_patch.setattr(
            wire_v3,
            "reencode_formal_variant_batch_v3",
            record_single,
        )
        single_outputs = tuple(
            adapter.make_variant_envelopes(
                envelopes=adapter.envelopes,
                decoded=decoded,
                plan=plan,
            )
            for plan in plans
        )

    assert tuple(single_schedules) == bulk_schedules
    bulk_roots = tuple(
        wire_v3._variant_schedule_root_v3(schedule)  # type: ignore[attr-defined]
        for schedule in bulk_schedules
    )
    single_roots = tuple(
        wire_v3._variant_schedule_root_v3(schedule)  # type: ignore[attr-defined]
        for schedule in single_schedules
    )
    assert bulk_roots == single_roots
    assert len(set(bulk_roots)) == 80
    assert bulk.batches == single_outputs


def test_wire_v3_rich32_executes_exact_distinct_32_32_16_schedules(
    monkeypatch: pytest.MonkeyPatch,
):
    import hegel_machine.phase2b_formal_trusted_wire_v3 as wire_v3
    import test_phase2b_formal_trusted_wire_v3 as wire_fixture

    cases = tuple(wire_fixture.source_case(index) for index in range(32))
    global_catalog, catalog_binding = wire_fixture.catalog(cases)
    success = wire_v3.build_formal_trusted_wire_batch_v3(
        source_cases=cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=catalog_binding,
        run_id=b"Q" * 32,
        key_sources=wire_fixture.keys(),
        key_source_receipt=wire_fixture.receipt(),
    )
    assert type(success) is wire_v3.FormalTrustedWireBuildSuccessV3
    adapter = FormalWireV3AdapterV1(
        success,
        run_id=b"Q" * 32,
        key_sources=wire_fixture.keys(),
        source_cases=cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=catalog_binding,
    )
    development = _authority()
    labels = tuple(
        FormalAuthorityLabelInputV1(
            row.row_index,
            row.family,
            row.binding,
            row.scale,
            row.answerability,
            row.joint_decision_class,
            row.case_type,
        )
        for row in development.rows
    )
    authority = issue_formal_wire_v3_audit_authority_v1(success, labels)
    binding = adapter.bind_private_authority(authority, adapter.envelopes)
    assert binding.alignment_issuance_verified
    assert binding.label_row_binding_verified

    suite_schedule_calls: list[tuple[object, ...]] = []
    original_variant_suite = wire_v3.reencode_formal_variant_suite_v3

    def record_variant_suite(
        *, build_success: object, schedules: tuple[object, ...]
    ) -> object:
        suite_schedule_calls.append(schedules)
        return original_variant_suite(
            build_success=build_success,
            schedules=schedules,
        )

    monkeypatch.setattr(
        wire_v3,
        "reencode_formal_variant_suite_v3",
        record_variant_suite,
    )
    decoded = tuple(adapter.decode_envelope(row) for row in adapter.envelopes)
    for item in decoded:
        auditor_module._validate_decoded(item, adapter)
    recognizer = _ConstantRecognizer()
    adapter_pins, _ = auditor_module._capture_adapter_execution_pins(
        adapter,
        recognizer,
    )
    input_root = auditor_module._input_root(adapter.envelopes)
    fork_input_snapshot_root = auditor_module._invariance_input_snapshot_root(
        adapter.envelopes,
        decoded,
        adapter,
        recognizer,
        input_root,
        adapter_pins,
    )
    result = auditor_module._run_invariance_core(
        adapter.envelopes,
        decoded,
        adapter,
        recognizer,
        input_root,
        fork_input_snapshot_root,
        adapter_pins,
    )
    assert result.invariant_execution_root.startswith(
        "phase2b_formal_covert_invariant_execution_"
    )
    assert result.exact_invariant_comparisons == 32 * 80 * 5
    assert recognizer.calls == 81
    assert len(suite_schedule_calls) == 1
    schedules = suite_schedule_calls[0]
    assert len(schedules) == 80
    assert result.consumer_decode_count == 32 * 80
    assert result.variant_suite_unique_envelope_object_count <= 32 * 49
    assert result.variant_suite_resident_object_graph_bytes_observed > 0

    identity_case_order = success.public_batch.envelope_ids
    uuid_schedules = schedules[:32]
    case_schedules = schedules[32:64]
    observation_schedules = schedules[64:]
    baseline_observation_orders = tuple(
        sorted(
            (
                envelope.envelope_id,
                tuple(
                    row["observation_id"]
                    for row in auditor_module._resolve(
                        item.payload, item.observation_array_paths[0]
                    )
                ),
            )
            for envelope, item in zip(
                success.public_batch.envelopes, decoded, strict=True
            )
        )
    )
    source_domain = {
        (leaf.namespace, leaf.public_uuid)
        for item in decoded
        for leaf in item.namespace_leaves
    }
    assert all(
        schedule.case_order_envelope_ids == identity_case_order
        and tuple(
            (row.envelope_id, row.observation_ids)
            for row in schedule.observation_orders
        )
        == baseline_observation_orders
        and {
            (row.namespace, row.source_uuid) for row in schedule.uuid_renamings
        }
        == source_domain
        and {row.namespace for row in schedule.uuid_renamings}
        == set(auditor_module.RENAMING_NAMESPACES)
        and len({row.target_uuid for row in schedule.uuid_renamings})
        == len(source_domain)
        for schedule in uuid_schedules
    )
    assert len(
        {
            tuple(
                (row.namespace, row.source_uuid, row.target_uuid)
                for row in schedule.uuid_renamings
            )
            for schedule in uuid_schedules
        }
    ) == 32
    assert all(
        not schedule.uuid_renamings
        and tuple(
            (row.envelope_id, row.observation_ids)
            for row in schedule.observation_orders
        )
        == baseline_observation_orders
        and schedule.case_order_envelope_ids != identity_case_order
        for schedule in case_schedules
    )
    assert len(
        {schedule.case_order_envelope_ids for schedule in case_schedules}
    ) == 32
    assert all(
        schedule.case_order_envelope_ids == identity_case_order
        and not schedule.uuid_renamings
        and len(schedule.observation_orders) == 32
        and tuple(
            (row.envelope_id, row.observation_ids)
            for row in schedule.observation_orders
        )
        != baseline_observation_orders
        for schedule in observation_schedules
    )
    assert len(
        {
            tuple(
                (row.envelope_id, row.observation_ids)
                for row in schedule.observation_orders
            )
            for schedule in observation_schedules
        }
    ) == 16


def test_raw_statistics_workers_require_plan_and_evidence_before_any_fork(
    monkeypatch: pytest.MonkeyPatch,
):
    fork_calls = 0

    def forbidden_fork() -> int:
        nonlocal fork_calls
        fork_calls += 1
        raise AssertionError("missing final-preflight inputs reached fork")

    monkeypatch.setattr(auditor_module.os, "fork", forbidden_fork)
    partition = auditor_module._partition(("a", "b", "a", "b"))
    _, masks = auditor_module._label_masks(("x", "y", "x", "y"))
    with pytest.raises(TypeError, match="preflight_plan"):
        auditor_module._run_raw_statistics_workers(
            (partition,),
            tuple(masks for _ in auditor_module.TARGETS),
            tuple((masks,) * 10_000 for _ in auditor_module.TARGETS),
            "/unreached",
            -1,
            1_048_576,
            1,
            time.monotonic_ns() + 10_000_000_000,
        )
    assert fork_calls == 0


def test_raw_statistics_workers_emit_five_closed_artifacts_and_reap_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    partition = auditor_module._partition(("a", "b", "a", "b"))
    _, masks = auditor_module._label_masks(("x", "y", "x", "y"))
    permutation_masks = tuple(masks for _ in range(10_000))
    with monkeypatch.context() as local:
        local.setattr(auditor_module, "_linux_effective_cpu_capacity", lambda: 1)
        projection, deadline_ns, invariant_plan = _phase_plan_test_inputs(
            4,
            (partition,),
        )
        one_worker_plan = auditor_module._resource_plan(
            4,
            (partition,),
            (2, 2, 2, 2, 2),
            adapter_retained_graph_bytes_bound=2 * 1_048_576,
            partition_projection=projection,
            partition_preparation_deadline_ns=deadline_ns,
            invariance_plan=invariant_plan,
        )
    parallel_partitions = tuple(
        auditor_module._Partition(
            masks=(index + 1, ((1 << 32) - 1) ^ (index + 1)),
            counts=(16, 16),
        )
        for index in range(100)
    )
    with monkeypatch.context() as local:
        local.setattr(auditor_module, "_linux_effective_cpu_capacity", lambda: 32)
        projection, deadline_ns, invariant_plan = _phase_plan_test_inputs(
            32,
            parallel_partitions,
        )
        five_worker_plan = auditor_module._resource_plan(
            32,
            parallel_partitions,
            (256, 256, 256, 256, 256),
            adapter_retained_graph_bytes_bound=2 * 1_048_576,
            partition_projection=projection,
            partition_preparation_deadline_ns=deadline_ns,
            invariance_plan=invariant_plan,
        )
    assert one_worker_plan.maximum_parallel_workers == 1
    assert five_worker_plan.maximum_parallel_workers == 5
    plans = {1: one_worker_plan, 5: five_worker_plan}
    preflight_slots: list[
        tuple[int, object, list[object | None]]
    ] = []

    def run_raw(
        directory: str,
        directory_fd: int,
        maximum_workers: int,
        deadline_ns: int,
    ) -> tuple[tuple[int, int, int], ...]:
        slot: list[object | None] = [None]
        plan = plans[maximum_workers]
        preflight_slots.append((maximum_workers, plan, slot))
        return auditor_module._run_raw_statistics_workers(
            (partition,),
            tuple(masks for _ in auditor_module.TARGETS),
            tuple(permutation_masks for _ in auditor_module.TARGETS),
            directory,
            directory_fd,
            1_048_576,
            maximum_workers,
            deadline_ns,
            preflight_plan=plan,
            preflight_evidence_slot=slot,
        )

    def workspace(label: str) -> tuple[str, int, os.stat_result]:
        directory = tempfile.mkdtemp(
            prefix=f"hegel-formal-covert-stats-{label}-",
            dir=auditor_module.ARTIFACT_TEMP_ROOT,
        )
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        return directory, descriptor, os.fstat(descriptor)

    directory, directory_fd, directory_stat = workspace("success")
    try:
        executions = run_raw(
            directory,
            directory_fd,
            1,
            time.monotonic_ns() + 10_000_000_000,
        )
        assert tuple(item[0] for item in executions) == tuple(range(5))
        assert len({item[1] for item in executions}) == 1
        assert set(os.listdir(directory_fd)) == {
            f"target-{index}.json" for index in range(5)
        }
    finally:
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)

    directory, directory_fd, directory_stat = workspace("sigchld-ignore")
    try:
        with monkeypatch.context() as signal_patch:
            real_getsignal = signal.getsignal
            signal_patch.setattr(
                signal,
                "getsignal",
                lambda signum: (
                    signal.SIG_IGN
                    if signum == signal.SIGCHLD
                    else real_getsignal(signum)
                ),
            )
            with pytest.raises(
                FormalCovertAuditRejectionV1,
                match="FAIL_STATISTICAL_FORK_SAFETY",
            ):
                run_raw(
                    directory,
                    directory_fd,
                    1,
                    time.monotonic_ns() + 10_000_000_000,
                )
        sigchld_reads = 0
        with monkeypatch.context() as signal_patch:
            real_getsignal = signal.getsignal

            def flip_after_block(signum: int) -> object:
                nonlocal sigchld_reads
                if signum != signal.SIGCHLD:
                    return real_getsignal(signum)
                sigchld_reads += 1
                return signal.SIG_DFL if sigchld_reads == 1 else signal.SIG_IGN

            signal_patch.setattr(signal, "getsignal", flip_after_block)
            with pytest.raises(
                FormalCovertAuditRejectionV1,
                match="FAIL_STATISTICAL_FORK_SAFETY",
            ):
                run_raw(
                    directory,
                    directory_fd,
                    1,
                    time.monotonic_ns() + 10_000_000_000,
                )
        assert sigchld_reads == 2
        assert not os.listdir(directory_fd)
    finally:
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)

    sigchld_calls: list[int] = []
    prior_sigchld = signal.getsignal(signal.SIGCHLD)

    def observe_sigchld(signum: int, _frame: object) -> None:
        sigchld_calls.append(signum)

    signal.signal(signal.SIGCHLD, observe_sigchld)
    directory, directory_fd, directory_stat = workspace("sigchld-owner")
    try:
        run_raw(
            directory,
            directory_fd,
            1,
            time.monotonic_ns() + 10_000_000_000,
        )
    finally:
        signal.signal(signal.SIGCHLD, prior_sigchld)
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)
    assert sigchld_calls and set(sigchld_calls) == {signal.SIGCHLD}

    directory, directory_fd, directory_stat = workspace("five-worker-success")
    try:
        executions = run_raw(
            directory,
            directory_fd,
            5,
            time.monotonic_ns() + 10_000_000_000,
        )
        assert tuple(item[0] for item in executions) == tuple(range(5))
        assert len({item[1] for item in executions}) == 5
        assert all(item[2] > 0 for item in executions)
    finally:
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)

    real_fork = os.fork

    def hang(*_args: object) -> object:
        time.sleep(30.0)
        raise AssertionError("unreachable")

    partial_fork_pids: list[int] = []
    fork_calls = 0

    def fail_third_fork() -> int:
        nonlocal fork_calls
        fork_calls += 1
        if fork_calls == 3:
            raise OSError("injected third-fork failure")
        pid = real_fork()
        if pid > 0:
            partial_fork_pids.append(pid)
        return pid

    directory, directory_fd, directory_stat = workspace("partial-fork")
    try:
        with monkeypatch.context() as local:
            local.setattr(auditor_module, "_evaluate_target_statistics", hang)
            local.setattr(auditor_module.os, "fork", fail_third_fork)
            with pytest.raises(OSError, match="third-fork"):
                run_raw(
                    directory,
                    directory_fd,
                    5,
                    time.monotonic_ns() + 5_000_000_000,
                )
    finally:
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)
    assert len(partial_fork_pids) == 2
    assert all(not Path(f"/proc/{pid}").exists() for pid in partial_fork_pids)

    real_pidfd_open = os.pidfd_open
    partial_pidfd_pids: list[int] = []
    partial_pidfds: list[int] = []
    pidfd_calls = 0

    def fail_second_pidfd(pid: int, flags: int) -> int:
        nonlocal pidfd_calls
        pidfd_calls += 1
        partial_pidfd_pids.append(pid)
        if pidfd_calls == 2:
            raise OSError("injected second-pidfd failure")
        descriptor = real_pidfd_open(pid, flags)
        partial_pidfds.append(descriptor)
        return descriptor

    directory, directory_fd, directory_stat = workspace("partial-pidfd")
    try:
        with monkeypatch.context() as local:
            local.setattr(auditor_module, "_evaluate_target_statistics", hang)
            local.setattr(auditor_module.os, "pidfd_open", fail_second_pidfd)
            with pytest.raises(OSError, match="second-pidfd"):
                run_raw(
                    directory,
                    directory_fd,
                    5,
                    time.monotonic_ns() + 5_000_000_000,
                )
    finally:
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)
    assert len(partial_pidfd_pids) == 2
    assert all(not Path(f"/proc/{pid}").exists() for pid in partial_pidfd_pids)
    assert partial_pidfds
    for descriptor in partial_pidfds:
        with pytest.raises(OSError):
            os.fstat(descriptor)

    captured_pids: list[int] = []

    def capture_pidfd(pid: int, flags: int) -> int:
        captured_pids.append(pid)
        return real_pidfd_open(pid, flags)

    monkeypatch.setattr(auditor_module.os, "pidfd_open", capture_pidfd)

    def system_exit(*_args: object) -> object:
        raise SystemExit("injected worker BaseException")

    monkeypatch.setattr(
        auditor_module,
        "_evaluate_target_statistics",
        system_exit,
    )
    directory, directory_fd, directory_stat = workspace("baseexception")
    try:
        with pytest.raises(RuntimeError, match="exited without all assigned"):
            run_raw(
                directory,
                directory_fd,
                1,
                time.monotonic_ns() + 5_000_000_000,
            )
    finally:
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)
    assert captured_pids and all(
        not Path(f"/proc/{pid}").exists() for pid in captured_pids
    )

    monkeypatch.setattr(auditor_module, "_evaluate_target_statistics", hang)
    captured_pids.clear()
    directory, directory_fd, directory_stat = workspace("timeout")
    started = time.monotonic()
    try:
        with pytest.raises(
            FormalCovertAuditRejectionV1,
            match="FAIL_INVARIANCE_RUNTIME",
        ):
            run_raw(
                directory,
                directory_fd,
                1,
                time.monotonic_ns() + 50_000_000,
            )
    finally:
        auditor_module._cleanup_statistics_artifact_directory(
            directory,
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(directory_fd)
    assert time.monotonic() - started < 10.0
    assert captured_pids and all(
        not Path(f"/proc/{pid}").exists() for pid in captured_pids
    )
    assert len(preflight_slots) == 9
    assert {
        index
        for index, (_workers, _plan, slot) in enumerate(preflight_slots)
        if slot[0] is None
    } == {1, 2}
    for _workers, plan, slot in preflight_slots:
        phase_host = slot[0]
        if phase_host is None:
            continue
        assert type(phase_host) is auditor_module._PhaseHostMemoryPreflightV1
        auditor_module._validate_statistics_phase_preflight_against_resource_plan(
            plan,
            phase_host.resource_preflight,
        )


def test_raw_invariance_wait4_nonzero_noartifact_timeout_and_descendant_reap(
    tmp_path: Path,
):
    def spawn(exit_code: int, *, ignore_term: bool = False) -> tuple[int, int]:
        pid = os.fork()
        if pid == 0:
            os.setsid()
            if ignore_term:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                while True:
                    time.sleep(1.0)
            os._exit(exit_code)
        return pid, os.pidfd_open(pid, 0)

    ordinary_pid, ordinary_pidfd = spawn(0)
    try:
        status, peak, _ = auditor_module._wait_raw_invariance_child(
            ordinary_pid,
            2.0,
            ordinary_pidfd,
            [False],
        )
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        assert peak > 0
    finally:
        os.close(ordinary_pidfd)

    base_pid, base_pidfd = spawn(70)
    try:
        status, _, _ = auditor_module._wait_raw_invariance_child(
            base_pid,
            2.0,
            base_pidfd,
            [False],
        )
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 70
    finally:
        os.close(base_pidfd)

    artifact_directory = tmp_path / "hegel-formal-covert-invariant-noartifact"
    artifact_directory.mkdir()
    directory_fd = os.open(
        str(artifact_directory),
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    directory_stat = os.fstat(directory_fd)
    try:
        with pytest.raises(FileNotFoundError):
            auditor_module._read_invariant_worker_artifact(
                "invariant-result.json",
                str(artifact_directory),
                directory_fd,
                directory_stat.st_dev,
                directory_stat.st_ino,
                1_048_576,
                os.getpid(),
                1,
            )
        auditor_module._cleanup_invariant_artifact_directory(
            str(artifact_directory),
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
    finally:
        os.close(directory_fd)

    hanging_pid, hanging_pidfd = spawn(0, ignore_term=True)
    started = time.monotonic()
    try:
        with pytest.raises(
            FormalCovertAuditRejectionV1,
            match="exceeded its absolute deadline",
        ):
            auditor_module._wait_raw_invariance_child(
                hanging_pid,
                0.05,
                hanging_pidfd,
                [False],
            )
        assert time.monotonic() - started < 8.0
    finally:
        os.close(hanging_pidfd)
    assert not Path(f"/proc/{hanging_pid}").exists()

    leader = os.fork()
    if leader == 0:
        os.setsid()
        descendant = os.fork()
        if descendant == 0:
            while True:
                time.sleep(1.0)
        os._exit(0)
    leader_pidfd = os.pidfd_open(leader, 0)
    try:
        with pytest.raises(
            FormalCovertAuditRejectionV1,
            match="left descendant processes",
        ):
            auditor_module._wait_raw_invariance_child(
                leader,
                2.0,
                leader_pidfd,
                [False],
            )
    finally:
        os.close(leader_pidfd)
    assert not auditor_module._linux_process_group_members(leader)


def test_raw_invariance_child_closes_high_fd_after_soft_limit_lowering():
    inherited_fd = os.open("/dev/null", os.O_RDONLY)
    high_fd = 512
    os.dup2(inherited_fd, high_fd)
    pid = os.fork()
    if pid == 0:
        os.setsid()
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
        auditor_module._close_inherited_fds_for_invariance_child()
        try:
            os.fstat(high_fd)
        except OSError:
            os._exit(0)
        os._exit(1)
    os.close(inherited_fd)
    os.close(high_fd)
    waited, status, _ = os.wait4(pid, 0)
    assert waited == pid
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def test_invariant_artifact_closed_schema_links_bounds_and_path_swap(
    tmp_path: Path,
):
    result = _valid_invariant_result()

    def workspace(name: str) -> tuple[Path, int, os.stat_result]:
        directory = Path(
            tempfile.mkdtemp(
                prefix=f"hegel-formal-covert-invariant-{name}-",
                dir="/tmp",
            )
        )
        auditor_module._write_invariant_child_artifact(
            result,
            str(directory / "invariant-result.json"),
            1_048_576,
        )
        descriptor = os.open(
            str(directory),
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        return directory, descriptor, os.fstat(descriptor)

    def read(directory: Path, descriptor: int, observed: os.stat_result) -> object:
        return auditor_module._read_invariant_worker_artifact(
            "invariant-result.json",
            str(directory),
            descriptor,
            observed.st_dev,
            observed.st_ino,
            1_048_576,
            os.getpid(),
            1,
        )

    valid_directory, valid_fd, valid_stat = workspace("valid")
    replay, artifact = read(valid_directory, valid_fd, valid_stat)
    assert replay == result
    assert artifact.sha256.startswith("sha256:")
    auditor_module._cleanup_invariant_artifact_directory(
        str(valid_directory), valid_fd, valid_stat.st_dev, valid_stat.st_ino
    )
    os.close(valid_fd)

    mutations = {
        "duplicate": b'{"artifact_schema":"x","artifact_schema":"x"}',
        "unknown": auditor_module.canonicalize_rfc8785_v1(
            {
                "artifact_schema": auditor_module.INVARIANCE_WORKER_ARTIFACT_SCHEMA,
                "unknown": True,
            }
        ),
        "missing": auditor_module.canonicalize_rfc8785_v1(
            {"artifact_schema": auditor_module.INVARIANCE_WORKER_ARTIFACT_SCHEMA}
        ),
    }
    for name, payload in mutations.items():
        directory, descriptor, observed = workspace(name)
        (directory / "invariant-result.json").write_bytes(payload)
        with pytest.raises(ValueError, match="canonical|closed schema"):
            read(directory, descriptor, observed)
        auditor_module._cleanup_invariant_artifact_directory(
            str(directory), descriptor, observed.st_dev, observed.st_ino
        )
        os.close(descriptor)

    typed_directory, typed_fd, typed_stat = workspace("type")
    typed_path = typed_directory / "invariant-result.json"
    typed_body = json.loads(typed_path.read_bytes())
    typed_body["consumer_decode_count_decimal"] = 80
    typed_path.write_bytes(auditor_module.canonicalize_rfc8785_v1(typed_body))
    with pytest.raises(ValueError, match="decimal integer encoding"):
        read(typed_directory, typed_fd, typed_stat)
    auditor_module._cleanup_invariant_artifact_directory(
        str(typed_directory), typed_fd, typed_stat.st_dev, typed_stat.st_ino
    )
    os.close(typed_fd)

    oversized_directory, oversized_fd, oversized_stat = workspace("oversized")
    (oversized_directory / "invariant-result.json").write_bytes(
        b"x" * (1_048_576 + 1)
    )
    with pytest.raises(ValueError, match="bounded regular link"):
        read(oversized_directory, oversized_fd, oversized_stat)
    auditor_module._cleanup_invariant_artifact_directory(
        str(oversized_directory),
        oversized_fd,
        oversized_stat.st_dev,
        oversized_stat.st_ino,
    )
    os.close(oversized_fd)

    hardlink_directory, hardlink_fd, hardlink_stat = workspace("hardlink")
    os.link(
        hardlink_directory / "invariant-result.json",
        hardlink_directory / "second-link",
    )
    with pytest.raises(ValueError, match="bounded regular link"):
        read(hardlink_directory, hardlink_fd, hardlink_stat)
    auditor_module._cleanup_invariant_artifact_directory(
        str(hardlink_directory),
        hardlink_fd,
        hardlink_stat.st_dev,
        hardlink_stat.st_ino,
    )
    os.close(hardlink_fd)

    symlink_directory, symlink_fd, symlink_stat = workspace("symlink")
    artifact_path = symlink_directory / "invariant-result.json"
    victim_file = tmp_path / "artifact-victim"
    victim_file.write_bytes(b"sentinel")
    artifact_path.unlink()
    artifact_path.symlink_to(victim_file)
    with pytest.raises(ValueError, match="bounded regular link"):
        read(symlink_directory, symlink_fd, symlink_stat)
    auditor_module._cleanup_invariant_artifact_directory(
        str(symlink_directory),
        symlink_fd,
        symlink_stat.st_dev,
        symlink_stat.st_ino,
    )
    os.close(symlink_fd)
    assert victim_file.read_bytes() == b"sentinel"

    swapped_directory, swapped_fd, swapped_stat = workspace("path-swap")
    retained_directory = swapped_directory.with_name(
        swapped_directory.name + "-retained"
    )
    swapped_directory.rename(retained_directory)
    victim_directory = tmp_path / "victim-directory"
    victim_directory.mkdir()
    victim_sentinel = victim_directory / "must-survive"
    victim_sentinel.write_bytes(b"untouched")
    swapped_directory.symlink_to(victim_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="directory path was replaced"):
        read(swapped_directory, swapped_fd, swapped_stat)
    with pytest.raises(ValueError, match="replaced by a symlink"):
        auditor_module._cleanup_invariant_artifact_directory(
            str(swapped_directory),
            swapped_fd,
            swapped_stat.st_dev,
            swapped_stat.st_ino,
        )
    os.close(swapped_fd)
    retained_directory.rmdir()
    assert victim_sentinel.read_bytes() == b"untouched"


def test_child_callback_copyreg_and_decoder_mutation_never_reaches_parent(
    tmp_path: Path,
):
    authority = _authority()
    wire = _SyntheticWire()
    sentinel = tmp_path / "parent-child-mutation-sentinel"
    recognizer = _ChildCopyregRecognizer(wire, sentinel)
    original_decoder = auditor_module.decode_and_require_canonical_rfc8785_v1
    original_reducer = copyreg.dispatch_table.get(
        auditor_module._InvariantExecutionResultV1
    )
    envelopes = tuple(_envelope(_payload()) for _ in authority.rows)
    receipt = run_formal_covert_audit_v1(
        envelopes,
        authority,
        wire,
        recognizer,
    )
    assert receipt.disposition is FormalCovertDisposition.PASS
    assert not sentinel.exists()
    assert auditor_module.decode_and_require_canonical_rfc8785_v1 is original_decoder
    assert copyreg.dispatch_table.get(
        auditor_module._InvariantExecutionResultV1
    ) is original_reducer
    assert receipt.invariant_runtime_evidence.parent_side_adapter_sandboxed is False


def test_adapter_pin_drift_native_task_drift_and_forged_recognizer_outputs_reject(
    monkeypatch: pytest.MonkeyPatch,
):
    authority = _authority()
    envelopes = tuple(_envelope(_payload()) for _ in authority.rows)
    drifting_wire = _DriftingWire()
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_ADAPTER_EXECUTION_PIN_DRIFT",
    ):
        run_formal_covert_audit_v1(
            envelopes,
            authority,
            drifting_wire,
            _SemanticRecognizer(drifting_wire),
        )

    wire = _SyntheticWire()
    monkeypatch.setattr(
        auditor_module,
        "_linux_process_task_ids",
        lambda: (threading.get_native_id(), threading.get_native_id() + 1),
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_INVARIANCE_FORK_SAFETY",
    ):
        run_formal_covert_audit_v1(
            envelopes,
            authority,
            wire,
            _SemanticRecognizer(wire),
        )
    monkeypatch.undo()

    binding = FormalBindingV1(
        _UUIDS["role_candidate_id"],
        _UUIDS["entity_id"],
    )
    forged_rows = (
        _forged_decision(binding=(binding, binding)),
        _forged_decision(
            binding=tuple(
                FormalBindingV1(
                    f"{index:08x}-0000-4000-8000-{index:012x}",
                    f"{index + 1:08x}-0000-4000-8000-{index + 1:012x}",
                )
                for index in range(257)
            )
        ),
        _forged_decision(
            scale_set=tuple(
                f"{index:08x}-0000-4000-8000-{index:012x}"
                for index in range(257)
            )
        ),
        _forged_decision(binding=type("TupleSubclass", (tuple,), {})()),
    )
    for forged in forged_rows:
        with pytest.raises(
            FormalCovertAuditRejectionV1,
            match="FAIL_RECOGNIZER_OUTPUT_REPLAY",
        ):
            auditor_module._run_recognizer(
                _ForgedDecisionRecognizer(forged),
                (bytes(65_536),),
            )


def test_resource_plan_is_b_plus_one_cgroup_aware_and_worker_artifact_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
):
    monkeypatch.setattr(auditor_module.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(
        auditor_module.os,
        "sched_getaffinity",
        lambda _: {0, 1, 2},
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_cgroup_cpu_quota_capacity",
        lambda: 2,
    )
    assert auditor_module._linux_effective_cpu_capacity() == 2

    def affinity_unavailable(_: int) -> set[int]:
        raise OSError("injected affinity read failure")

    monkeypatch.setattr(
        auditor_module.os,
        "sched_getaffinity",
        affinity_unavailable,
    )
    assert auditor_module._linux_effective_cpu_capacity() == 1

    partition = auditor_module._partition(("a", "b", "a", "b"))
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: 64 * 1_048_576,
    )
    projection, deadline_ns, invariant_plan = _phase_plan_test_inputs(
        4,
        (partition,),
    )
    plan = auditor_module._resource_plan(
        4,
        (partition,),
        (2, 2, 2, 2, 2),
        adapter_retained_graph_bytes_bound=2 * 1_048_576,
        partition_projection=projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan,
    )
    assert plan.estimated_intersection_work_units == (
        10_001 * plan.intersections_per_exchangeable_draw
    )
    assert plan.bias_center_canonicalization_work_units == (
        plan.unique_statistical_partitions
        * 5
        * (10_001 * 17 + 17**2)
    )
    assert plan.maximum_shuffle_rejection_draws == (
        plan.maximum_label_shuffle_randbelow_calls * 4_096
    )
    assert plan.maximum_parallel_workers == 1
    required = (
        plan.estimated_peak_rss_bytes
        + plan.host_available_memory_safety_margin_bytes
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_system_available_memory_bytes",
        lambda: required + 1_000_000,
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_cgroup_memory_remaining_bytes",
        lambda: required - 1,
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_STATISTICAL_HOST_MEMORY",
    ):
        auditor_module._preflight_host_memory(plan)
    monkeypatch.setattr(
        auditor_module,
        "_linux_cgroup_memory_remaining_bytes",
        lambda: None,
    )
    memory = auditor_module._preflight_host_memory(plan)
    assert memory.effective_available_bytes == required + 1_000_000

    monkeypatch.setattr(
        auditor_module,
        "_linux_effective_cpu_capacity",
        lambda: 32,
    )
    parallel_partitions = tuple(
        auditor_module._Partition(
            masks=(index + 1, ((1 << 32) - 1) ^ (index + 1)),
            counts=(16, 16),
        )
        for index in range(100)
    )
    projection, deadline_ns, invariant_plan = _phase_plan_test_inputs(
        32,
        parallel_partitions,
    )
    five_worker_plan = auditor_module._resource_plan(
        32,
        parallel_partitions,
        (256, 256, 256, 256, 256),
        adapter_retained_graph_bytes_bound=2 * 1_048_576,
        partition_projection=projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan,
    )
    assert five_worker_plan.estimated_intersection_work_units >= (
        auditor_module.PARALLEL_STATISTICS_WORK_THRESHOLD
    )
    assert five_worker_plan.maximum_parallel_workers == 5
    assert five_worker_plan.admissible is True

    directory = tmp_path / "hegel-formal-covert-stats-kat"
    directory.mkdir()
    directory_fd = os.open(
        str(directory),
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    directory_stat = os.fstat(directory_fd)
    rows = (((0.0, 0.0, 0.0, 10_000, 10_000),), 0.0, 10_000)
    auditor_module._write_statistics_worker_artifact(
        0,
        rows,
        str(directory),
        directory_fd,
        1_048_576,
    )
    decoded, artifact = auditor_module._read_statistics_worker_artifact(
        0,
        str(directory),
        directory_fd,
        directory_stat.st_dev,
        directory_stat.st_ino,
        1_048_576,
        1,
        os.getpid(),
        1,
    )
    assert decoded == rows
    with open(artifact.path, "r+b") as handle:
        handle.seek(0)
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="canonical closed JSON"):
        auditor_module._read_statistics_worker_artifact(
            0,
            str(directory),
            directory_fd,
            directory_stat.st_dev,
            directory_stat.st_ino,
            1_048_576,
            1,
            os.getpid(),
            1,
        )
    auditor_module._cleanup_statistics_artifact_directory(
        str(directory),
        directory_fd,
        directory_stat.st_dev,
        directory_stat.st_ino,
    )
    os.close(directory_fd)


def test_phase_split_exact_diagnostic_formulas_and_preflight_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    parent_rss = 3_036_942_336
    envelope_decode = 1_194_949_240
    adapter = 1_012_034_804
    variant_fixed = 268_435_456
    variant_per_row = 3_252_224
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: parent_rss,
    )
    invariant_plan = auditor_module._invariance_resource_plan(
        1_024,
        envelope_and_decode_bytes_bound=envelope_decode,
        adapter_retained_graph_bytes_bound=adapter,
        variant_suite_resident_bytes_per_row_bound=variant_per_row,
        variant_suite_resident_fixed_bytes_bound=variant_fixed,
    )
    assert invariant_plan.invariance_variant_suite_resident_bytes_bound == (
        3_598_712_832
    )
    assert invariant_plan.invariance_recognizer_live_output_bytes_bound == (
        1_006_632_960
    )
    assert invariant_plan.estimated_invariance_peak_rss_bytes == 10_680_279_040
    profile = json.loads(
        (
            Path(__file__).parents[1]
            / "config"
            / "phase2b_formal_covert_auditor_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert profile["resource_boundary"]["invariance_rss_formula"] == (
        invariant_plan.invariance_rss_formula
    )
    assert (
        invariant_plan.maximum_invariance_estimated_rss_bytes
        - invariant_plan.estimated_invariance_peak_rss_bytes
        == 2_204_622_848
    )

    statistics_static = 19_200_000 + adapter + 130_784_960 + 64 * 1_048_576
    statistics_tail = 1_429_904_896
    statistics_preflight = auditor_module._PhaseResourcePreflightEvidenceV1(
        phase="STATISTICS",
        resource_plan_root=(
            "phase2b_formal_covert_statistics_resource_plan_" + "0" * 64
        ),
        parent_current_rss_bytes_at_preflight=parent_rss,
        static_parent_bytes_bound=statistics_static,
        fork_parent_rss_multiplier=2,
        additional_bytes_bound=statistics_tail,
        projected_aggregate_rss_bytes_at_preflight=7_503_789_568,
        maximum_estimated_rss_bytes=auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES,
        host_available_memory_safety_margin_bytes=(
            auditor_module.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ),
    )
    statistics_preflight.__post_init__()
    assert statistics_static == 1_229_128_628
    assert (
        auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
        - statistics_preflight.projected_aggregate_rss_bytes_at_preflight
        == 5_381_112_320
    )
    with pytest.raises(ValueError, match="projection formula drift"):
        replace(
            statistics_preflight,
            projected_aggregate_rss_bytes_at_preflight=(
                statistics_preflight.projected_aggregate_rss_bytes_at_preflight
                + 1
            ),
        )
    with pytest.raises(ValueError, match="plan root drift"):
        replace(
            statistics_preflight,
            resource_plan_root=invariant_plan.plan_root,
        )
    with pytest.raises(ValueError, match="exceeds frozen cap"):
        replace(
            statistics_preflight,
            parent_current_rss_bytes_at_preflight=(
                auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
            ),
            projected_aggregate_rss_bytes_at_preflight=(
                auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES * 2
                + statistics_tail
            ),
        )
    statistics_root = (
        "phase2b_formal_covert_statistics_resource_plan_" + "1" * 64
    )
    static_dominant = auditor_module._PhaseResourcePreflightEvidenceV1(
        phase="STATISTICS",
        resource_plan_root=statistics_root,
        parent_current_rss_bytes_at_preflight=100,
        static_parent_bytes_bound=200,
        fork_parent_rss_multiplier=2,
        additional_bytes_bound=3,
        projected_aggregate_rss_bytes_at_preflight=403,
        maximum_estimated_rss_bytes=auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES,
        host_available_memory_safety_margin_bytes=(
            auditor_module.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ),
    )
    current_dominant = replace(
        static_dominant,
        parent_current_rss_bytes_at_preflight=200,
        static_parent_bytes_bound=100,
    )
    assert static_dominant.projected_aggregate_rss_bytes_at_preflight == 403
    assert current_dominant.projected_aggregate_rss_bytes_at_preflight == 403
    exact_half_cap = auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES // 2
    at_cap = replace(
        static_dominant,
        parent_current_rss_bytes_at_preflight=exact_half_cap,
        static_parent_bytes_bound=1,
        additional_bytes_bound=0,
        projected_aggregate_rss_bytes_at_preflight=(
            auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
        ),
    )
    assert at_cap.projected_aggregate_rss_bytes_at_preflight == (
        auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES
    )
    with pytest.raises(ValueError, match="exceeds frozen cap"):
        replace(
            at_cap,
            additional_bytes_bound=1,
            projected_aggregate_rss_bytes_at_preflight=(
                auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES + 1
            ),
        )


def test_partition_measurement_is_single_and_stats_rss_sample_follows_it(
    monkeypatch: pytest.MonkeyPatch,
):
    deadline_ns = (
        time.monotonic_ns()
        + auditor_module.PARTITION_PREPARATION_TIMEOUT_SECONDS * 1_000_000_000
    )
    columns = (
        auditor_module._FeatureColumn(("alias/a",), ("a", "b", "a", "b")),
    )
    projection = auditor_module._partition_allocation_projection(
        4,
        columns,
        deadline_ns,
    )
    original_deep = auditor_module._deep_object_graph_bytes
    with monkeypatch.context() as local:
        local.setattr(
            auditor_module,
            "_deep_object_graph_bytes",
            lambda _value: (_ for _ in ()).throw(
                AssertionError("partition materialization must not deep-scan")
            ),
        )
        partitions = auditor_module._materialize_partitions_bounded(
            columns,
            projection,
            deadline_ns,
        )
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: 64 * 1_048_576,
    )
    invariant_plan = auditor_module._invariance_resource_plan(
        4,
        envelope_and_decode_bytes_bound=4 * auditor_module.ENVELOPE_BYTES * 2,
        adapter_retained_graph_bytes_bound=2 * 1_048_576,
        variant_suite_resident_bytes_per_row_bound=(
            auditor_module.ENVELOPE_BYTES
            + auditor_module.ATOMIC_VARIANT_SUITE_REFERENCE_BYTES_PER_ENVELOPE
        ),
        variant_suite_resident_fixed_bytes_bound=1_048_576,
    )
    events: list[str] = []

    def measured(value: object) -> int:
        events.append("partition_deep")
        return original_deep(value)

    def sampled() -> int:
        events.append("statistics_rss")
        return 64 * 1_048_576

    monkeypatch.setattr(auditor_module, "_deep_object_graph_bytes", measured)
    monkeypatch.setattr(auditor_module, "_linux_current_rss_bytes", sampled)
    plan = auditor_module._resource_plan(
        4,
        partitions,
        (2, 2, 2, 2, 2),
        adapter_retained_graph_bytes_bound=2 * 1_048_576,
        partition_projection=projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan,
    )
    with pytest.raises(ValueError, match="invariant phase root drift"):
        replace(
            plan,
            invariance_resource_plan_root=(
                "phase2b_formal_covert_invariant_resource_plan_" + "0" * 64
            ),
        )
    assert events == ["partition_deep", "statistics_rss"]
    assert plan.partition_matrix_bytes_bound == (
        auditor_module.DEEP_SIZE_SAFETY_MULTIPLIER
        * original_deep(partitions)
    )


def test_phase_preflight_records_same_current_projection_and_rejects_growth(
    monkeypatch: pytest.MonkeyPatch,
):
    partition = auditor_module._partition(("a", "b", "a", "b"))
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: 64 * 1_048_576,
    )
    projection, deadline_ns, invariant_plan = _phase_plan_test_inputs(
        4,
        (partition,),
    )
    plan = auditor_module._resource_plan(
        4,
        (partition,),
        (2, 2, 2, 2, 2),
        adapter_retained_graph_bytes_bound=2 * 1_048_576,
        partition_projection=projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan,
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: 65 * 1_048_576,
    )
    invariant_plan_2 = auditor_module._invariance_resource_plan(
        4,
        envelope_and_decode_bytes_bound=4 * auditor_module.ENVELOPE_BYTES * 2,
        adapter_retained_graph_bytes_bound=2 * 1_048_576,
        variant_suite_resident_bytes_per_row_bound=(
            auditor_module.ENVELOPE_BYTES
            + auditor_module.ATOMIC_VARIANT_SUITE_REFERENCE_BYTES_PER_ENVELOPE
        ),
        variant_suite_resident_fixed_bytes_bound=1_048_576,
    )
    plan_2 = auditor_module._resource_plan(
        4,
        (partition,),
        (2, 2, 2, 2, 2),
        adapter_retained_graph_bytes_bound=2 * 1_048_576,
        partition_projection=projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan_2,
    )
    assert plan_2.resource_plan_root != plan.resource_plan_root
    assert (
        plan_2.invariance_resource_plan_root
        != plan.invariance_resource_plan_root
    )
    current = 128 * 1_048_576
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: current,
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_system_available_memory_bytes",
        lambda: 100 * 1_073_741_824,
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_cgroup_memory_remaining_bytes",
        lambda: None,
    )
    evidence = auditor_module._preflight_host_memory(plan)
    invariant_evidence = auditor_module._preflight_invariance_host_memory(
        invariant_plan
    )
    auditor_module._validate_statistics_phase_preflight_against_resource_plan(
        plan,
        evidence.resource_preflight,
    )
    auditor_module._validate_invariant_phase_preflight_against_resource_plan(
        plan,
        invariant_evidence.resource_preflight,
    )
    with pytest.raises(ValueError, match="statistics.*splice drift"):
        auditor_module._validate_statistics_phase_preflight_against_resource_plan(
            plan,
            replace(
                evidence.resource_preflight,
                resource_plan_root=plan_2.resource_plan_root,
            ),
        )
    with pytest.raises(ValueError, match="invariant.*splice drift"):
        auditor_module._validate_invariant_phase_preflight_against_resource_plan(
            plan,
            replace(
                invariant_evidence.resource_preflight,
                resource_plan_root=plan_2.invariance_resource_plan_root,
            ),
        )
    assert evidence.resource_preflight.parent_current_rss_bytes_at_preflight == current
    assert evidence.resource_preflight.projected_aggregate_rss_bytes_at_preflight == (
        max(current, evidence.resource_preflight.static_parent_bytes_bound)
        * evidence.resource_preflight.fork_parent_rss_multiplier
        + evidence.resource_preflight.additional_bytes_bound
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: auditor_module.MAXIMUM_ESTIMATED_RSS_BYTES,
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_STATISTICAL_RESOURCE_PLAN",
    ):
        auditor_module._preflight_host_memory(plan)
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_INVARIANCE_RESOURCE_PLAN",
    ):
        auditor_module._preflight_invariance_host_memory(invariant_plan)


def test_phase_order_and_final_preflight_are_adjacent_to_raw_fork_inputs():
    audit_source = getsource(
        auditor_module._run_formal_covert_audit_with_reservation_v1
    )
    assert audit_source.index("sufficiency = tuple(") < audit_source.index(
        "invariant_plan = _invariance_resource_plan("
    ) < audit_source.index("_run_invariance(") < audit_source.index(
        "manifest, columns = _extract_feature_columns("
    )
    assert audit_source.count("_preflight_feature_extraction_memory(") == 2
    assert audit_source.index("_run_invariance(") < audit_source.rindex(
        "_preflight_feature_extraction_memory("
    ) < audit_source.index("manifest, columns = _extract_feature_columns(")

    invariant_source = getsource(auditor_module._run_invariance)
    assert invariant_source.index(
        "fork_input_snapshot_root = _invariance_input_snapshot_root("
    ) < invariant_source.index(
        "host_memory = _preflight_invariance_host_memory(plan)"
    ) < invariant_source.index("child_pid = os.fork()")

    statistics_source = getsource(auditor_module._run_raw_statistics_workers)
    assert statistics_source.index("active: list[dict[str, object]]") < (
        statistics_source.index(
            "preflight_evidence_slot[0] = _preflight_host_memory(preflight_plan)"
        )
    ) < statistics_source.index("pid = os.fork()")


def test_parent_preallocation_gates_charge_current_rss_without_fork_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    envelopes = tuple(_envelope(_payload()) for _ in range(4))
    wire = _SyntheticWire()
    decoded = tuple(wire.decode_envelope(item) for item in envelopes)
    feature_projection = auditor_module._feature_cardinality_projection(decoded)
    envelope_decode = auditor_module._measured_envelope_decode_bytes_bound(
        len(envelopes),
        envelopes,
        decoded,
    )
    result_bound = auditor_module._parent_result_working_bytes_bound(
        feature_projection.projected_concrete_feature_count
    )
    artifact_bound = len(auditor_module.TARGETS) * (
        feature_projection.projected_concrete_feature_count
        * auditor_module.WORKER_RESULT_ARTIFACT_BYTES_PER_FEATURE
        + auditor_module.WORKER_RESULT_ARTIFACT_FIXED_BYTES
    )
    feature_tail = (
        feature_projection.projected_feature_matrix_bytes_bound
        + result_bound
        + artifact_bound
    )
    state = {
        "current": 1,
        "available": envelope_decode
        + feature_tail
        + auditor_module.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES,
    }
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: state["current"],
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_system_available_memory_bytes",
        lambda: state["available"],
    )
    monkeypatch.setattr(
        auditor_module,
        "_linux_cgroup_memory_remaining_bytes",
        lambda: None,
    )
    feature_gate = auditor_module._preflight_feature_extraction_memory(
        feature_projection,
        envelope_decode,
    )
    assert type(feature_gate) is auditor_module._HostMemoryPreflightV1
    assert not hasattr(feature_gate, "resource_preflight")
    state["current"] = envelope_decode + 1_048_576
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_FEATURE_PROJECTION_HOST_MEMORY",
    ):
        auditor_module._preflight_feature_extraction_memory(
            feature_projection,
            envelope_decode,
        )

    deadline_ns = (
        time.monotonic_ns()
        + auditor_module.PARTITION_PREPARATION_TIMEOUT_SECONDS * 1_000_000_000
    )
    columns = (
        auditor_module._FeatureColumn(("alias/a",), ("a", "b", "a", "b")),
    )
    partition_projection = auditor_module._partition_allocation_projection(
        4,
        columns,
        deadline_ns,
    )
    feature_matrix = 1_048_576
    partition_result = auditor_module._parent_result_working_bytes_bound(1)
    resident_static = feature_matrix + envelope_decode
    state["current"] = 1
    state["available"] = (
        resident_static
        + partition_projection.projected_partition_matrix_bytes_bound
        + partition_result
        + auditor_module.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
    )
    partition_gate = auditor_module._preflight_partition_allocation_memory(
        feature_matrix_bytes_bound=feature_matrix,
        envelope_and_decode_bytes_bound=envelope_decode,
        partition_projection=partition_projection,
        concrete_features=1,
    )
    assert type(partition_gate) is auditor_module._HostMemoryPreflightV1
    assert not hasattr(partition_gate, "resource_preflight")
    state["current"] = resident_static + 1_048_576
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PARTITION_PROJECTION_HOST_MEMORY",
    ):
        auditor_module._preflight_partition_allocation_memory(
            feature_matrix_bytes_bound=feature_matrix,
            envelope_and_decode_bytes_bound=envelope_decode,
            partition_projection=partition_projection,
            concrete_features=1,
        )


def test_cgroup_hierarchy_uses_same_level_pairs_and_tightest_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    mountinfo = (
        "40 30 0:40 / /custom/cpu\\040mount rw - cgroup cgroup rw,cpu,cpuacct\n"
        "41 30 0:41 / /custom/unified rw - cgroup2 cgroup2 rw,nsdelegate\n"
    )
    assert auditor_module._parse_cgroup_mount_roots(
        mountinfo, "cpu"
    ) == ((Path("/custom/cpu mount"), Path("/")),)
    assert auditor_module._parse_cgroup_mount_roots(
        mountinfo, None
    ) == ((Path("/custom/unified"), Path("/")),)
    with monkeypatch.context() as local:
        local.setattr(
            auditor_module,
            "_self_cgroup_membership_path",
            lambda _controller: Path("/docker/id/child"),
        )
        local.setattr(
            auditor_module,
            "_cgroup_mount_roots",
            lambda _controller: (
                (Path("/sys/fs/cgroup"), Path("/docker/id")),
            ),
        )
        assert auditor_module._cgroup_ancestor_directories(None) == (
            Path("/sys/fs/cgroup/child"),
            Path("/sys/fs/cgroup"),
        )

    v2_leaf = tmp_path / "v2" / "leaf"
    v2_parent = tmp_path / "v2"
    v2_leaf.mkdir(parents=True)
    (v2_leaf / "cpu.max").write_text("max 100000\n", encoding="ascii")
    (v2_parent / "cpu.max").write_text("200000 100000\n", encoding="ascii")
    (v2_leaf / "memory.max").write_text("max\n", encoding="ascii")
    (v2_leaf / "memory.current").write_text("100\n", encoding="ascii")
    (v2_parent / "memory.max").write_text("1000\n", encoding="ascii")
    (v2_parent / "memory.current").write_text("600\n", encoding="ascii")
    monkeypatch.setattr(
        auditor_module,
        "_cgroup_ancestor_directories",
        lambda controller: (v2_leaf, v2_parent) if controller is None else (),
    )
    assert auditor_module._linux_cgroup_cpu_quota_capacity() == 2
    assert auditor_module._linux_cgroup_memory_remaining_bytes() == 400

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "memory.max").write_text("1000\n", encoding="ascii")
    monkeypatch.setattr(
        auditor_module,
        "_cgroup_ancestor_directories",
        lambda controller: (partial,) if controller is None else (),
    )
    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_PLATFORM_RESOURCE_SEMANTICS",
    ):
        auditor_module._linux_cgroup_memory_remaining_bytes()

    v1_leaf = tmp_path / "v1" / "leaf"
    v1_parent = tmp_path / "v1"
    v1_leaf.mkdir(parents=True)
    (v1_leaf / "cpu.cfs_quota_us").write_text("-1\n", encoding="ascii")
    (v1_leaf / "cpu.cfs_period_us").write_text("100000\n", encoding="ascii")
    (v1_parent / "cpu.cfs_quota_us").write_text("300000\n", encoding="ascii")
    (v1_parent / "cpu.cfs_period_us").write_text("100000\n", encoding="ascii")
    (v1_leaf / "memory.limit_in_bytes").write_text(
        str(1 << 62), encoding="ascii"
    )
    (v1_leaf / "memory.usage_in_bytes").write_text("100\n", encoding="ascii")
    (v1_parent / "memory.limit_in_bytes").write_text("2000\n", encoding="ascii")
    (v1_parent / "memory.usage_in_bytes").write_text("1250\n", encoding="ascii")
    monkeypatch.setattr(
        auditor_module,
        "_cgroup_ancestor_directories",
        lambda controller: () if controller is None else (v1_leaf, v1_parent),
    )
    assert auditor_module._linux_cgroup_cpu_quota_capacity() == 3
    assert auditor_module._linux_cgroup_memory_remaining_bytes() == 750


def test_synthetic_rich1024_formula_is_admissible_not_r05_capacity_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    # This freezes arithmetic reachability only.  The partitions/counts and
    # memory observations below are synthetic formula inputs; this test is not
    # rich-graph extraction, 81-run execution, 10k-statistics, deadline, RSS,
    # R04, or R05 qualification evidence.
    rows = 1_024
    concrete_features = 16_863
    unique_count = 7_991
    unique = tuple(
        auditor_module._Partition(
            masks=(index + 1, ((1 << rows) - 1) ^ (index + 1)),
            counts=(512, 512),
        )
        for index in range(unique_count)
    )
    partitions = unique + (unique[0],) * (concrete_features - unique_count)
    feature_floor = rows * concrete_features * 72
    envelope_floor = rows * auditor_module.ENVELOPE_BYTES * 2
    partition_bound = (
        auditor_module.DEEP_SIZE_SAFETY_MULTIPLIER
        * auditor_module._deep_object_graph_bytes(partitions)
    )
    monkeypatch.setattr(auditor_module, "_linux_effective_cpu_capacity", lambda: 32)
    monkeypatch.setattr(
        auditor_module,
        "_linux_current_rss_bytes",
        lambda: 128 * 1_048_576,
    )
    adapter_bound = auditor_module.MAXIMUM_WIRE_ADAPTER_RETAINED_GRAPH_BYTES
    variant_per_row = (
        auditor_module.WIRE_V3_ATOMIC_SUITE_UNIQUE_BATCH_EQUIVALENTS
        * auditor_module.ENVELOPE_BYTES
        + 80
        * auditor_module.ATOMIC_VARIANT_SUITE_REFERENCE_BYTES_PER_ENVELOPE
    )
    projection, deadline_ns, invariant_plan = _phase_plan_test_inputs(
        rows,
        partitions,
        envelope_and_decode_bytes_bound=envelope_floor,
        adapter_retained_graph_bytes_bound=adapter_bound,
        variant_suite_resident_bytes_per_row_bound=variant_per_row,
        variant_suite_resident_fixed_bytes_bound=(
            auditor_module.ATOMIC_VARIANT_SUITE_FIXED_BYTES_BOUND
        ),
    )
    plan = auditor_module._resource_plan(
        rows,
        partitions,
        (2, 2, 2, 2, 100),
        preextraction_feature_matrix_bytes_bound=feature_floor,
        feature_matrix_bytes_bound=feature_floor,
        envelope_and_decode_bytes_bound=envelope_floor,
        adapter_retained_graph_bytes_bound=adapter_bound,
        partition_projection=projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan,
    )
    plan.__post_init__()
    assert plan.rows == 1_024
    assert plan.concrete_features == 16_863
    assert plan.unique_statistical_partitions == 7_991
    assert plan.hypotheses == 84_315
    assert plan.maximum_parallel_workers == 1
    assert plan.invariance_variant_suite_output_bytes_bound == (
        rows * 80 * auditor_module.ENVELOPE_BYTES
    )
    assert plan.invariance_variant_suite_resident_bytes_bound < (
        plan.invariance_variant_suite_output_bytes_bound
    )
    assert plan.estimated_invariance_peak_rss_bytes <= (
        plan.maximum_invariance_estimated_rss_bytes
    )
    assert plan.estimated_peak_rss_bytes <= plan.maximum_estimated_rss_bytes
    assert plan.estimated_total_statistical_work_units <= (
        plan.maximum_total_statistical_work_units
    )
    assert plan.admissible is True


def test_statistics_artifact_closed_schema_link_bounds_and_path_swap(
    tmp_path: Path,
):
    rows = (((0.0, 0.0, 0.0, 10_000, 10_000),), 0.0, 10_000)

    def workspace(label: str) -> tuple[Path, int, os.stat_result]:
        path = Path(
            tempfile.mkdtemp(
                prefix=f"hegel-formal-covert-stats-{label}-",
                dir=auditor_module.ARTIFACT_TEMP_ROOT,
            )
        )
        descriptor = os.open(
            str(path),
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        return path, descriptor, os.fstat(descriptor)

    def read(
        directory: Path,
        descriptor: int,
        directory_stat: os.stat_result,
        maximum_bytes: int = 1_048_576,
    ) -> object:
        return auditor_module._read_statistics_worker_artifact(
            0,
            str(directory),
            descriptor,
            directory_stat.st_dev,
            directory_stat.st_ino,
            maximum_bytes,
            1,
            os.getpid(),
            1,
        )

    def write_valid(directory: Path, descriptor: int) -> Path:
        auditor_module._write_statistics_worker_artifact(
            0,
            rows,
            str(directory),
            descriptor,
            1_048_576,
        )
        return directory / "target-0.json"

    def cleanup(
        directory: Path,
        descriptor: int,
        directory_stat: os.stat_result,
    ) -> None:
        auditor_module._cleanup_statistics_artifact_directory(
            str(directory),
            descriptor,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
        os.close(descriptor)

    for label, mutate, expected in (
        (
            "unknown-field",
            lambda body: {**body, "unknown": "forbidden"},
            "closed schema drift",
        ),
        (
            "missing-field",
            lambda body: {
                key: value
                for key, value in body.items()
                if key != "artifact_schema"
            },
            "closed schema drift",
        ),
        (
            "float-spelling",
            lambda body: {
                **body,
                "rows": [["0x0.00p+0", *body["rows"][0][1:]]],
            },
            "float.hex canonicalization drift",
        ),
    ):
        directory, descriptor, directory_stat = workspace(label)
        artifact_path = write_valid(directory, descriptor)
        body = auditor_module.decode_and_require_canonical_rfc8785_v1(
            artifact_path.read_bytes()
        )
        artifact_path.write_bytes(
            auditor_module.canonicalize_rfc8785_v1(mutate(body))
        )
        with pytest.raises(ValueError, match=expected):
            read(directory, descriptor, directory_stat)
        cleanup(directory, descriptor, directory_stat)

    directory, descriptor, directory_stat = workspace("duplicate-key")
    (directory / "target-0.json").write_bytes(
        b'{"artifact_schema":"x","artifact_schema":"x"}'
    )
    with pytest.raises(ValueError, match="canonical closed JSON"):
        read(directory, descriptor, directory_stat)
    cleanup(directory, descriptor, directory_stat)

    directory, descriptor, directory_stat = workspace("tamper")
    artifact_path = write_valid(directory, descriptor)
    payload = artifact_path.read_bytes()
    artifact_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(ValueError, match="canonical closed JSON"):
        read(directory, descriptor, directory_stat)
    cleanup(directory, descriptor, directory_stat)

    directory, descriptor, directory_stat = workspace("oversize")
    (directory / "target-0.json").write_bytes(b"x" * 1_025)
    with pytest.raises(ValueError, match="byte/type/link drift"):
        read(directory, descriptor, directory_stat, maximum_bytes=1_024)
    cleanup(directory, descriptor, directory_stat)

    directory, descriptor, directory_stat = workspace("hardlink")
    source = directory / "source"
    source.write_bytes(b"source")
    os.link(source, directory / "target-0.json")
    with pytest.raises(ValueError, match="byte/type/link drift"):
        read(directory, descriptor, directory_stat)
    cleanup(directory, descriptor, directory_stat)

    directory, descriptor, directory_stat = workspace("symlink")
    victim = tmp_path / "statistics-artifact-victim"
    victim.write_bytes(b"sentinel")
    (directory / "target-0.json").symlink_to(victim)
    with pytest.raises(ValueError, match="byte/type/link drift"):
        read(directory, descriptor, directory_stat)
    cleanup(directory, descriptor, directory_stat)
    assert victim.read_bytes() == b"sentinel"

    directory, descriptor, directory_stat = workspace("path-swap")
    write_valid(directory, descriptor)
    retained = directory.with_name(directory.name + "-retained")
    directory.rename(retained)
    victim_directory = tmp_path / "statistics-victim-directory"
    victim_directory.mkdir()
    sentinel = victim_directory / "must-survive"
    sentinel.write_bytes(b"untouched")
    directory.symlink_to(victim_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="directory path was replaced"):
        read(directory, descriptor, directory_stat)
    with pytest.raises(ValueError, match="replaced by a symlink"):
        auditor_module._cleanup_statistics_artifact_directory(
            str(directory),
            descriptor,
            directory_stat.st_dev,
            directory_stat.st_ino,
        )
    os.close(descriptor)
    retained.rmdir()
    assert sentinel.read_bytes() == b"untouched"


def test_private_artifact_directory_partial_acquisition_cleans_every_resource(
    monkeypatch: pytest.MonkeyPatch,
):
    real_mkdtemp = tempfile.mkdtemp
    real_lstat = os.lstat
    real_open = os.open
    real_fstat = os.fstat
    real_close = os.close
    real_rmdir = os.rmdir

    captured_paths: list[str] = []

    def capture_mkdtemp(*args: object, **kwargs: object) -> str:
        path = real_mkdtemp(*args, **kwargs)
        captured_paths.append(path)
        return path

    with monkeypatch.context() as local:
        calls = 0

        def fail_first_lstat(path: object, *args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected lstat failure")
            return real_lstat(path, *args, **kwargs)

        local.setattr(auditor_module.tempfile, "mkdtemp", capture_mkdtemp)
        local.setattr(auditor_module.os, "lstat", fail_first_lstat)
        with pytest.raises(OSError, match="lstat failure"):
            auditor_module._acquire_private_artifact_directory(
                prefix="hegel-formal-covert-stats-"
            )
    assert captured_paths and not Path(captured_paths[-1]).exists()

    with monkeypatch.context() as local:
        local.setattr(auditor_module.tempfile, "mkdtemp", capture_mkdtemp)
        local.setattr(
            auditor_module.os,
            "open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected open failure")
            ),
        )
        with pytest.raises(OSError, match="open failure"):
            auditor_module._acquire_private_artifact_directory(
                prefix="hegel-formal-covert-invariant-"
            )
    assert not Path(captured_paths[-1]).exists()

    captured_fd = -1
    with monkeypatch.context() as local:
        def capture_open(*args: object, **kwargs: object) -> int:
            nonlocal captured_fd
            captured_fd = real_open(*args, **kwargs)
            return captured_fd

        def fail_owned_fstat(fd: int) -> os.stat_result:
            if fd == captured_fd:
                raise OSError("injected fstat failure")
            return real_fstat(fd)

        local.setattr(auditor_module.tempfile, "mkdtemp", capture_mkdtemp)
        local.setattr(auditor_module.os, "open", capture_open)
        local.setattr(auditor_module.os, "fstat", fail_owned_fstat)
        with pytest.raises(OSError, match="fstat failure"):
            auditor_module._acquire_private_artifact_directory(
                prefix="hegel-formal-covert-stats-"
            )
    with pytest.raises(OSError):
        real_fstat(captured_fd)
    assert not Path(captured_paths[-1]).exists()

    captured_fd = -1
    with monkeypatch.context() as local:
        def capture_open_for_close(*args: object, **kwargs: object) -> int:
            nonlocal captured_fd
            captured_fd = real_open(*args, **kwargs)
            return captured_fd

        def fail_fstat_for_close(fd: int) -> os.stat_result:
            if fd == captured_fd:
                raise OSError("primary fstat failure")
            return real_fstat(fd)

        def close_then_raise(fd: int) -> None:
            real_close(fd)
            raise OSError("injected close failure")

        local.setattr(auditor_module.tempfile, "mkdtemp", capture_mkdtemp)
        local.setattr(auditor_module.os, "open", capture_open_for_close)
        local.setattr(auditor_module.os, "fstat", fail_fstat_for_close)
        local.setattr(auditor_module.os, "close", close_then_raise)
        with pytest.raises(OSError, match="primary fstat failure") as caught:
            auditor_module._acquire_private_artifact_directory(
                prefix="hegel-formal-covert-stats-"
            )
        assert any("close failure" in note for note in caught.value.__notes__)
    assert not Path(captured_paths[-1]).exists()

    with monkeypatch.context() as local:
        def fail_open_again(*_args: object, **_kwargs: object) -> int:
            raise OSError("primary open failure")

        def rmdir_then_raise(path: object, *args: object, **kwargs: object) -> None:
            real_rmdir(path, *args, **kwargs)
            raise OSError("injected rmdir failure")

        local.setattr(auditor_module.tempfile, "mkdtemp", capture_mkdtemp)
        local.setattr(auditor_module.os, "open", fail_open_again)
        local.setattr(auditor_module.os, "rmdir", rmdir_then_raise)
        with pytest.raises(OSError, match="primary open failure") as caught:
            auditor_module._acquire_private_artifact_directory(
                prefix="hegel-formal-covert-invariant-"
            )
        assert any("rmdir failure" in note for note in caught.value.__notes__)
    assert not Path(captured_paths[-1]).exists()


def test_shuffle_rejection_sampling_has_a_hard_draw_cap():
    class RejectingStream(auditor_module._Sha256Stream):
        def _u64(self) -> int:
            return (1 << 64) - 1

    with pytest.raises(
        FormalCovertAuditRejectionV1,
        match="FAIL_SHUFFLE_REJECTION_BOUND",
    ):
        RejectingStream(b"bounded-shuffle-kat").randbelow(3)


def test_one_intersection_pass_is_bit_identical_to_frozen_two_loop_oracle():
    partitions = (
        auditor_module._partition(("a", "a", "b", "b", "c", "c")),
        auditor_module._partition(("u", "v", "u", "v", "u", "v")),
    )
    _, observed = auditor_module._label_masks(("x", "x", "y", "y", "x", "y"))
    patterns = (
        ("x", "y", "x", "y", "x", "y"),
        ("y", "x", "y", "x", "y", "x"),
        ("x", "x", "y", "y", "x", "y"),
    )
    permutation_masks = tuple(
        auditor_module._label_masks(patterns[index % len(patterns)])[1]
        for index in range(10_000)
    )
    expected = tuple(
        _old_two_loop_partition_oracle(
            partition,
            observed,
            permutation_masks,
        )
        for partition in partitions
    )
    observed_result = auditor_module._evaluate_target_statistics(
        partitions,
        observed,
        permutation_masks,
    )
    rows, observed_target_max, target_exceedance = observed_result
    observed_statistics = tuple(
        max(
            max(0.0, row[0] - row[2]) / 0.02,
            max(0.0, row[1]) / 0.05,
        )
        for row in expected
    )
    expected_target_max = max(observed_statistics)
    permutation_target_max: list[float] = []
    feature_localization_counts = [0] * len(partitions)
    for masks in permutation_masks:
        per_feature = []
        for partition, row in zip(partitions, expected, strict=True):
            nmi, advantage = auditor_module._metrics_from_masks(
                partition,
                masks,
                {},
            )
            per_feature.append(
                max(
                    max(0.0, nmi - row[2]) / 0.02,
                    max(0.0, advantage) / 0.05,
                )
            )
        maximum = max(per_feature)
        permutation_target_max.append(maximum)
        for index, observed_statistic in enumerate(observed_statistics):
            if maximum >= observed_statistic:
                feature_localization_counts[index] += 1
    expected_target_exceedance = sum(
        value >= expected_target_max for value in permutation_target_max
    )
    assert rows == tuple(
        row + (localization_count,)
        for row, localization_count in zip(
            expected,
            feature_localization_counts,
            strict=True,
        )
    )
    assert target_exceedance == expected_target_exceedance
    assert observed_target_max == expected_target_max
