from __future__ import annotations

import copy
from pathlib import Path

import pytest

from assumption_agent.benchmarks.paper_protocol import (
    PORTABLE_TASK_CAPABILITY_PROTOCOL_VERSION,
    TASK_INPUT_CLOSURE_PROTOCOL_VERSION,
    TYPED_SELECTION_PROTOCOL_VERSION,
    TYPED_SELECTION_PROTOCOL_VERSIONS,
    PaperProtocol,
)
from assumption_agent.benchmarks.prewarm import (
    DEVELOPMENT_PREWARM_VERSION,
    TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
    development_prewarm_version_for_protocol,
)
from assumption_agent.benchmarks.skilllearn_experiment import (
    _task_input_closure_policy_for_execution,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnPrebuiltImageCache,
)
from assumption_agent.benchmarks.task_input_closure import (
    TASK_INPUT_CLOSURE_POLICY_VERSION,
)
from assumption_agent.benchmarks.task_input_freeze import (
    load_frozen_task_input_closure,
)
from assumption_agent.benchmarks.typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
V318_PROTOCOL = (
    ROOT
    / "manifests"
    / "skilllearn_paper_protocol_v3_18r1_ruoli_gpt54mini.json"
)
V319_PROTOCOL = (
    ROOT
    / "manifests"
    / "skilllearn_paper_protocol_v3_19_ruoli_gpt54mini.json"
)


def test_v319_changes_only_public_input_closure_contract() -> None:
    v318_protocol = PaperProtocol.read(V318_PROTOCOL)
    v319_protocol = PaperProtocol.read(V319_PROTOCOL)

    assert v318_protocol.validate_structure() == []
    assert v319_protocol.validate_structure() == []
    assert v318_protocol.payload["protocol_version"] == (
        TYPED_SELECTION_PROTOCOL_VERSION
    )
    assert v319_protocol.payload["protocol_version"] == (
        TASK_INPUT_CLOSURE_PROTOCOL_VERSION
    )
    assert v318_protocol.payload["execution"]["development_prewarm"] == (
        DEVELOPMENT_PREWARM_VERSION
    )
    assert v319_protocol.payload["execution"]["development_prewarm"] == (
        TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION
    )
    assert v319_protocol.payload["execution"][
        "task_input_closure_policy"
    ] == TASK_INPUT_CLOSURE_POLICY_VERSION
    assert v319_protocol.promotion_gate_spec == (
        v318_protocol.promotion_gate_spec
    )
    assert v319_protocol.codex_agent_execution_policy == (
        v318_protocol.codex_agent_execution_policy
    )

    v318 = copy.deepcopy(v318_protocol.payload)
    v319 = copy.deepcopy(v319_protocol.payload)
    for payload in (v318, v319):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    v319["execution"]["development_prewarm"] = v318["execution"][
        "development_prewarm"
    ]
    v319["execution"].pop("task_input_closure_policy")
    v319["execution"].pop("task_input_closure_source")
    assert v319 == v318
    assert development_prewarm_version_for_protocol("3.18.0") == (
        DEVELOPMENT_PREWARM_VERSION
    )
    assert development_prewarm_version_for_protocol("3.19.0") == (
        TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION
    )
    assert development_prewarm_version_for_protocol("3.20.0") == (
        TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION
    )


def test_v319_loads_the_exact_hash_only_preparation_ledger() -> None:
    protocol = PaperProtocol.read(V319_PROTOCOL)
    frozen = load_frozen_task_input_closure(
        protocol.payload,
        project_root=ROOT,
    )
    assert frozen is not None
    assert frozen.source["preparation_receipt_file_sha256"] == (
        "73c25f8edd17dc719aa016cc6ca81b2f7d759186455c7c1787d381620e1b8b06"
    )
    assert frozen.source["preparation_receipt_hash"] == (
        "8d1979e5476a81189296e56e808d824bb547e067251ee9dc27811a42d6771a16"
    )
    assert frozen.source["closure_ledger_hash"] == (
        "7ad7f44826b4b008575bfd2b449a2454333f2a6f6bbf65ef5ad4d59a509ac3a9"
    )
    assert len(frozen.ledger_by_item_hash) == 11
    assert sum(
        int(row["object_count"])
        for row in frozen.ledger_by_item_hash.values()
    ) > int(frozen.source["content_object_count"])


def test_v319_rejects_preparation_receipt_hash_drift() -> None:
    protocol = PaperProtocol.read(V319_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"]["task_input_closure_source"][
        "preparation_receipt_hash"
    ] = "0" * 64
    with pytest.raises(PermissionError, match="receipt contract"):
        load_frozen_task_input_closure(payload, project_root=ROOT)


@pytest.mark.parametrize("mutation", (None, "drifted"))
def test_v319_rejects_missing_or_drifted_closure_policy(
    mutation: str | None,
) -> None:
    protocol = PaperProtocol.read(V319_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    if mutation is None:
        payload["execution"].pop("task_input_closure_policy")
    else:
        payload["execution"]["task_input_closure_policy"] = mutation

    assert "task_input_closure_policy_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


def test_v319_rejects_missing_or_incomplete_closure_source() -> None:
    protocol = PaperProtocol.read(V319_PROTOCOL)
    missing = copy.deepcopy(protocol.payload)
    missing["execution"].pop("task_input_closure_source")
    assert "task_input_closure_source_missing" in PaperProtocol(
        protocol.path,
        missing,
    ).validate_structure()

    incomplete = copy.deepcopy(protocol.payload)
    incomplete["execution"]["task_input_closure_source"].pop(
        "closure_ledger_hash"
    )
    assert "task_input_closure_source_fields_mismatch" in PaperProtocol(
        protocol.path,
        incomplete,
    ).validate_structure()


def test_v318_remains_closure_free() -> None:
    protocol = PaperProtocol.read(V318_PROTOCOL)
    assert "task_input_closure_policy" not in protocol.payload["execution"]
    assert set(TYPED_SELECTION_PROTOCOL_VERSIONS) == {
        TYPED_SELECTION_PROTOCOL_VERSION,
        TASK_INPUT_CLOSURE_PROTOCOL_VERSION,
        PORTABLE_TASK_CAPABILITY_PROTOCOL_VERSION,
    }

    payload = copy.deepcopy(protocol.payload)
    payload["execution"]["task_input_closure_policy"] = (
        TASK_INPUT_CLOSURE_POLICY_VERSION
    )
    assert "task_input_closure_policy_unexpected" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


def test_v320_requires_portable_compiler_and_keeps_v319_closed() -> None:
    v319 = PaperProtocol.read(V319_PROTOCOL)
    portable = copy.deepcopy(v319.payload)
    portable["protocol_id"] = "portable-structure-probe"
    portable["protocol_version"] = (
        PORTABLE_TASK_CAPABILITY_PROTOCOL_VERSION
    )
    portable["execution"]["portable_capability_compiler_mode"] = (
        PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
    )
    assert PaperProtocol(v319.path, portable).validate_structure() == []

    missing = copy.deepcopy(portable)
    missing["execution"].pop("portable_capability_compiler_mode")
    assert "portable_capability_compiler_mode_mismatch" in PaperProtocol(
        v319.path,
        missing,
    ).validate_structure()

    legacy = copy.deepcopy(v319.payload)
    legacy["execution"]["portable_capability_compiler_mode"] = (
        PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
    )
    assert "portable_capability_compiler_mode_unexpected" in PaperProtocol(
        v319.path,
        legacy,
    ).validate_structure()


def test_experiment_enables_closure_cache_only_for_v319(
    tmp_path: Path,
) -> None:
    v318 = PaperProtocol.read(V318_PROTOCOL)
    v319 = PaperProtocol.read(V319_PROTOCOL)

    v318_policy = _task_input_closure_policy_for_execution(
        v318,
        v318.payload["execution"],
    )
    v319_policy = _task_input_closure_policy_for_execution(
        v319,
        v319.payload["execution"],
    )
    assert v318_policy is None
    assert v319_policy == TASK_INPUT_CLOSURE_POLICY_VERSION
    assert SkillLearnPrebuiltImageCache(
        tmp_path,
        task_input_closure_policy=v318_policy,
    ).task_input_closure_policy is None
    assert SkillLearnPrebuiltImageCache(
        tmp_path,
        task_input_closure_policy=v319_policy,
    ).task_input_closure_policy == TASK_INPUT_CLOSURE_POLICY_VERSION


def test_experiment_rejects_unfrozen_closure_policy() -> None:
    v318 = PaperProtocol.read(V318_PROTOCOL)
    with pytest.raises(
        ValueError,
        match="not permitted",
    ):
        _task_input_closure_policy_for_execution(
            v318,
            {
                **v318.payload["execution"],
                "task_input_closure_policy": (
                    TASK_INPUT_CLOSURE_POLICY_VERSION
                ),
            },
        )

    v319 = PaperProtocol.read(V319_PROTOCOL)
    with pytest.raises(ValueError, match="missing or drifted"):
        _task_input_closure_policy_for_execution(
            v319,
            {
                key: value
                for key, value in v319.payload["execution"].items()
                if key != "task_input_closure_policy"
            },
        )
