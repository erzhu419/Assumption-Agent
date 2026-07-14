from __future__ import annotations

import hashlib
import json
from pathlib import Path

from assumption_agent.benchmarks.runtime_profile_injection import (
    RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
    VerifiedRuntimeProfile,
    bind_verified_runtime_profile_prompt,
    build_runtime_profile_prompt_capsule,
)
from assumption_agent.benchmarks.typed_profile_injection_integration import (
    TYPED_PROFILE_INJECTION_INTEGRATION_VERSION,
    _fixed_tamper_probes,
    _read_preregistration,
    verify_existing_typed_profile_injection,
)
from assumption_agent.models import stable_hash
from assumption_agent.splits import SplitManifest


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "manifests"
    / "skilllearn_typed_profile_injection_integration_v1.json"
)


def _profile(seed: str) -> VerifiedRuntimeProfile:
    profile_bytes = (
        json.dumps(
            {
                "profile_version": "test-v1",
                "seed": seed,
                "raw_input_locator_persisted": False,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return VerifiedRuntimeProfile(
        metadata_hash=stable_hash({"metadata": seed}),
        item_id_hash=stable_hash({"item": seed}),
        role_spec_hash=stable_hash({"role": seed}),
        effect_receipt_hash=stable_hash({"effect": seed}),
        output_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        profile_bytes=profile_bytes,
    )


def _execution(seed: str):
    from assumption_agent.benchmarks.typed_profile_injection_integration import (
        _CanaryExecution,
    )

    instruction = f"Complete fixture {seed}."
    capsule = build_runtime_profile_prompt_capsule(
        request_hash=stable_hash({"request": seed}),
        context_hash=stable_hash({"context": seed}),
        source_receipt_hash=stable_hash({"source": seed}),
        typed_binding_set_hash=stable_hash({"binding": seed}),
        public_instruction_hash=stable_hash(
            {"public_instruction": instruction}
        ),
        profiles=(_profile(seed),),
    )
    before = 'codex exec --json -- "$(cat {instruction_file})"'
    bound = bind_verified_runtime_profile_prompt(
        capsule,
        container_readback=capsule.fragment_bytes,
        run_template=before,
        public_instruction=instruction,
    )
    return _CanaryExecution(
        row={"item_id_hash": stable_hash({"item": seed})},
        capsule=capsule,
        receipt=bound.receipt,
        run_template_before=before,
        run_template_after=bound.run_template,
        public_instruction=instruction,
    )


def test_fixed_runtime_profile_tamper_vector_is_closed() -> None:
    probes = _fixed_tamper_probes(
        (_execution("a"), _execution("b"), _execution("c"))
    )
    assert probes == {
        "profile_byte_mutation": True,
        "fragment_strip_or_mutation": True,
        "cross_item_capsule_swap": True,
        "source_effect_receipt_rebinding": True,
    }


def test_preregistration_binds_current_train_only_scope() -> None:
    project_root, preregistration = _read_preregistration(PREREGISTRATION)
    assert project_root == ROOT
    assert preregistration["integration_policy"] == (
        TYPED_PROFILE_INJECTION_INTEGRATION_VERSION
    )
    assert preregistration["decision_budget"] == 1
    assert preregistration["delivery_mode"] == (
        RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
    )
    manifest = SplitManifest.read(ROOT / preregistration["manifest"])
    ids = {row["item_id"] for row in preregistration["train_canaries"]}
    assert ids == {
        "stock-data-visualization-1",
        "temperature-simulation-2",
        "organize-messy-files-2",
    }
    assert ids <= set(manifest.train_ids)
    assert not ids.intersection(manifest.validation_ids)
    assert not ids.intersection(manifest.test_ids)
    assert {
        "assumption_agent/benchmarks/runtime_profile_injection.py",
        "assumption_agent/benchmarks/skilllearn_lifecycle.py",
        "assumption_agent/benchmarks/typed_profile_injection_integration.py",
        "tests/test_typed_profile_injection_integration.py",
    } <= set(preregistration["implementation_files"])


def test_canonical_formal_result_verifies_without_live_execution() -> None:
    report = verify_existing_typed_profile_injection(
        preregistration_path=PREREGISTRATION,
    )
    assert report["integration_passed"] is True
    assert report["exact_replay_verified"] is True
    assert all(report["acceptance"].values())
    assert report["runtime"]["canary_count"] == 3
    assert report["runtime"]["parallel_workers"] == 3
    assert report["runtime"]["model_invoked"] is False
    assert report["runtime"]["task_backend_run_task_invoked"] is False
    assert report["semantic_model_consumption_claimed"] is False
    assert report["task_utility_claimed"] is False
