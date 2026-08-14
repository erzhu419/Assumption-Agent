from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from types import MappingProxyType

import pytest

from hegel_machine import phase2b_c1_container_actor_profile_v1 as profile_module
from hegel_machine.phase2b_c1_container_actor_profile_v1 import (
    FAIL_PROFILE_BINDING,
    FAIL_PROFILE_CLAIM_BOUNDARY,
    FAIL_PROFILE_IO,
    FAIL_PROFILE_JSON,
    FAIL_PROFILE_SCHEMA,
    PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_DOMAIN,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_PREFIX,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA,
    PROFILE_PATH,
    Phase2BC1ContainerActorProfileErrorV1,
    Phase2BC1ContainerActorProfileSnapshotV1,
    load_phase2b_c1_container_actor_profile_v1,
    validate_phase2b_c1_container_actor_profile_v1,
)


_CONTENT_DOMAIN = b"HEGEL/PHASE2B/C1/CONTAINER_ACTOR_PROFILE/V1\x00"
_CONTENT_PREFIX = "phase2b_c1_container_actor_profile_v1_"
_AUTHORITY_REQUIREMENT_DOMAIN = bytes.fromhex(
    "484547454c2f504841534532422f43312f4f574e45525f41434345505445445f"
    "434f4e5441494e45525f415554484f524954595f524551554952454d454e542f"
    "563100"
)
_AUTHORITY_REQUIREMENT_PREFIX = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
)


def _fifo_load_worker(
    profile_path: str,
    repository_root: str,
    output: multiprocessing.Queue,
) -> None:
    try:
        load_phase2b_c1_container_actor_profile_v1(
            Path(profile_path),
            repository_root=Path(repository_root),
        )
    except Phase2BC1ContainerActorProfileErrorV1 as exc:
        output.put(exc.code)
    except BaseException as exc:  # pragma: no cover - failure visibility
        output.put(type(exc).__name__)
    else:
        output.put("ACCEPTED")


def _decoded_profile() -> dict[str, object]:
    value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _write_profile(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )
    return path


def _copy_bound_sources(root: Path) -> dict[str, Path]:
    decoded = _decoded_profile()
    copied: dict[str, Path] = {}
    for name, binding in decoded["base_control_bindings"].items():
        relative = Path(binding["path"])
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(profile_module.PROJECT_ROOT / relative, target)
        copied[name] = target
    return copied


def test_load_returns_content_addressed_static_snapshot() -> None:
    snapshot = load_phase2b_c1_container_actor_profile_v1()
    decoded = _decoded_profile()
    canonical = json.dumps(
        decoded,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected_content_digest = hashlib.sha256(
        _CONTENT_DOMAIN + canonical
    ).hexdigest()

    assert PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_DOMAIN == _CONTENT_DOMAIN
    assert PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_PREFIX == _CONTENT_PREFIX

    assert snapshot.schema_version == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA
    assert snapshot.profile_id == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID
    assert snapshot.authority_class == PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1
    assert snapshot.profile_content_id == (
        _CONTENT_PREFIX
        + expected_content_digest
    )
    assert snapshot.canonical_profile_sha256 == hashlib.sha256(canonical).hexdigest()
    assert snapshot.source_profile_sha256 == hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    assert snapshot.bound_source_sha256_verified is True


def test_load_returns_fresh_deeply_immutable_snapshots() -> None:
    first = load_phase2b_c1_container_actor_profile_v1()
    second = load_phase2b_c1_container_actor_profile_v1()

    assert first is not second
    assert first.profile is not second.profile
    assert isinstance(first.profile, MappingProxyType)
    assert first.profile == second.profile
    assert first.profile["actor_roles"] is not second.profile["actor_roles"]
    with pytest.raises(TypeError):
        first.profile["profile_id"] = "attacked"
    with pytest.raises(TypeError):
        first.profile["authority_disclosure"]["organizational_independence"] = True
    with pytest.raises(TypeError):
        first.profile["actor_roles"][0]["purpose_id"] = 9


def test_snapshot_constructor_is_private() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        Phase2BC1ContainerActorProfileSnapshotV1()


def test_exact_six_roles_graph_signature_locations_and_mount_owners() -> None:
    profile = load_phase2b_c1_container_actor_profile_v1().profile
    roles = profile["actor_roles"]
    assert tuple(row["purpose_id"] for row in roles) == (1, 2, 3, 4, 5, 6)
    assert tuple(row["role"] for row in roles) == (
        "GOVERNANCE_ANCHOR",
        "CUSTODIAN",
        "STORE_RECEIPT",
        "TIME_ATTESTOR",
        "RECOGNIZER",
        "SCORER_EVALUATOR",
    )
    checkpoint = profile["checkpoint_signature_policy"]
    assert checkpoint["required_graph_signature_count"] == 4
    assert checkpoint["required_graph_roles_in_order"] == tuple(
        row["role"] for row in roles[:4]
    )
    assert checkpoint["graph_signer_locations"] == (
        ("GOVERNANCE_ANCHOR", "checkpoint.checkpoint_signatures[0]"),
        ("CUSTODIAN", "checkpoint.checkpoint_signatures[1]"),
        ("STORE_RECEIPT", "store_receipt.detached_signature"),
        ("TIME_ATTESTOR", "time_token.detached_signature"),
    )
    operation = profile["execution_signature_policy"]
    assert operation["required_roles_in_order"] == ("RECOGNIZER", "SCORER_EVALUATOR")
    assert operation["required_signature_purposes_in_order"] == (
        "RECOGNIZER_OPERATION_RECEIPT",
        "SCORER_EVALUATION_REPORT",
    )
    assert operation["implemented_by_static_profile_loader"] is False
    assert roles[4]["checkpoint_signature_purpose"] is None
    assert roles[5]["checkpoint_signature_purpose"] is None
    assert "/custody" in roles[1]["exclusive_writable_mounts"]
    assert "/ledger" in roles[2]["exclusive_writable_mounts"]
    assert "/cas" in roles[2]["exclusive_writable_mounts"]
    assert "/prediction" in roles[4]["exclusive_writable_mounts"]
    assert "/score" in roles[5]["exclusive_writable_mounts"]


def test_owner_accepted_requirement_is_independently_content_addressed_and_versions_legacy() -> None:
    profile = load_phase2b_c1_container_actor_profile_v1().profile
    requirement = profile["owner_accepted_authority_requirement"]
    statements = json.dumps(
        requirement["statements"],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected = _AUTHORITY_REQUIREMENT_PREFIX + hashlib.sha256(
        _AUTHORITY_REQUIREMENT_DOMAIN + statements
    ).hexdigest()
    assert requirement["domain_hex"] == _AUTHORITY_REQUIREMENT_DOMAIN.hex()
    assert requirement["id_prefix"] == _AUTHORITY_REQUIREMENT_PREFIX
    assert requirement["requirement_id"] == expected
    assert requirement["legacy_external_authority_requirement_id"].endswith(
        "fea188566b28e0a8bf50ce0ae760a63e5ee8d0355a737fea30ed8cd8f05779e0"
    )
    assert requirement["legacy_external_candidate_verifier_alone_satisfies_requirement"] is False
    assert requirement["future_execution_registry_must_bind_requirement_id"] is True
    assert requirement["future_execution_registry_must_bind_profile_content_id"] is True


def test_actor_inputs_close_reveal_scoring_and_terminal_gate_evidence() -> None:
    roles = load_phase2b_c1_container_actor_profile_v1().profile["actor_roles"]
    by_role = {row["role"]: row for row in roles}
    assert {
        "PREDICTION_STORE_IF_ABSENT_RECEIPT",
        "SIGNED_PREDICTIONS_COMMITTED_TOKEN",
    } <= set(by_role["CUSTODIAN"]["read_only_input_classes"])
    assert {
        "PREDICTION_STORE_IF_ABSENT_RECEIPT",
        "SIGNED_REVEAL_TOKEN",
    } <= set(by_role["SCORER_EVALUATOR"]["read_only_input_classes"])
    assert {
        "EXECUTION_OPERATION_RECEIPTS",
        "TERMINAL_REPORT_STORE_RECEIPT",
        "SIGNED_LOGICAL_TERMINAL_TOKEN",
    } <= set(by_role["GOVERNANCE_ANCHOR"]["read_only_input_classes"])


def test_static_loader_keeps_every_actual_and_permanent_nonclaim_false() -> None:
    snapshot = load_phase2b_c1_container_actor_profile_v1()
    claim_policy = snapshot.profile["claim_policy"]
    assert all(
        value is False
        for value in claim_policy["actual_run_claims_required_false"].values()
    )
    assert all(
        value is False
        for value in claim_policy["permanent_nonclaims_required_false"].values()
    )
    assert snapshot.docker_invoked is False
    assert snapshot.keys_generated is False
    assert snapshot.signatures_generated is False
    assert snapshot.recognizer_executed is False
    assert snapshot.scoring_performed is False
    assert snapshot.c1_exit_evidence is False


def test_bound_phase3_sources_match_the_exact_profile_hashes() -> None:
    profile = load_phase2b_c1_container_actor_profile_v1().profile
    for binding in profile["base_control_bindings"].values():
        payload = (profile_module.PROJECT_ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding["sha256"]
    reuse = profile["reuse_boundary"]
    assert reuse["phase3_qualification_report_reusable_as_phase2b_evidence"] is False
    assert reuse["phase3_actor_keys_reusable"] is False
    assert reuse["phase3_actor_volumes_reusable"] is False
    assert reuse["fresh_phase2b_six_role_qualification_required"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"profile_id": "wrong"}),
        lambda value: value["scope"].reverse(),
        lambda value: value["authority_disclosure"].update(
            {"organizational_independence": True}
        ),
        lambda value: value["reuse_boundary"].update(
            {"phase3_qualification_report_reusable_as_phase2b_evidence": True}
        ),
        lambda value: value["owner_accepted_authority_requirement"].update(
            {"requirement_id": _AUTHORITY_REQUIREMENT_PREFIX + "0" * 64}
        ),
        lambda value: value["actor_roles"][4].update({"purpose_id": 4}),
        lambda value: value["actor_roles"][5]["forbidden_input_classes"].remove(
            "PRE_REVEAL_ANSWER_ACCESS"
        ),
        lambda value: value["checkpoint_signature_policy"].update(
            {"required_graph_signature_count": 6}
        ),
        lambda value: value["execution_signature_policy"][
            "signed_object_fields_in_order"
        ]["RECOGNIZER_OPERATION_RECEIPT"].remove("actor_profile_content_id"),
        lambda value: value["key_policy"].update({"fresh_key_count": 4}),
        lambda value: value["docker_control_policy"].update({"network": "bridge"}),
        lambda value: value["filesystem_policy"].update(
            {"hostile_owner_rollback_resistance_claimed": True}
        ),
        lambda value: value["chronology_policy"].update(
            {"wall_clock_accuracy_claimed": True}
        ),
    ],
    ids=(
        "profile-id",
        "scope-order",
        "organizational-claim",
        "phase3-evidence-reuse",
        "authority-requirement",
        "duplicate-purpose",
        "pre-reveal-access",
        "checkpoint-count",
        "execution-profile-binding",
        "key-count",
        "network",
        "rollback-claim",
        "wall-clock-claim",
    ),
)
def test_semantic_profile_drift_is_rejected(mutation) -> None:
    value = _decoded_profile()
    mutation(value)
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
        validate_phase2b_c1_container_actor_profile_v1(value)
    assert caught.value.code in {FAIL_PROFILE_SCHEMA, FAIL_PROFILE_CLAIM_BOUNDARY}


@pytest.mark.parametrize(
    "claim_group",
    ("actual_run_claims_required_false", "permanent_nonclaims_required_false"),
)
def test_false_claim_cannot_be_promoted_or_retyped(claim_group: str) -> None:
    for attacked in (True, 0):
        value = _decoded_profile()
        claims = value["claim_policy"][claim_group]
        first_name = next(iter(claims))
        claims[first_name] = attacked
        with pytest.raises(Phase2BC1ContainerActorProfileErrorV1):
            validate_phase2b_c1_container_actor_profile_v1(value)


def test_missing_or_extra_top_level_field_is_rejected() -> None:
    missing = _decoded_profile()
    missing.pop("chronology_policy")
    extra = _decoded_profile()
    extra["execution_authorized"] = False
    for value in (missing, extra):
        with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
            validate_phase2b_c1_container_actor_profile_v1(value)
        assert caught.value.code == FAIL_PROFILE_SCHEMA


def test_duplicate_json_key_and_nonfinite_or_float_number_are_rejected(tmp_path: Path) -> None:
    source = PROFILE_PATH.read_text(encoding="utf-8")
    duplicate = source.replace(
        '  "schema_version":',
        '  "schema_version": "duplicate",\n  "schema_version":',
        1,
    )
    float_number = source.replace('"fresh_key_count": 6', '"fresh_key_count": 6.0', 1)
    nonfinite = source.replace('"fresh_key_count": 6', '"fresh_key_count": NaN', 1)
    for index, payload in enumerate((duplicate, float_number, nonfinite)):
        attacked = tmp_path / f"attacked-{index}.json"
        attacked.write_text(payload, encoding="utf-8")
        with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
            load_phase2b_c1_container_actor_profile_v1(attacked)
        assert caught.value.code == FAIL_PROFILE_JSON


def test_symlink_profile_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "profile.json"
    link.symlink_to(PROFILE_PATH)
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
        load_phase2b_c1_container_actor_profile_v1(link)
    assert caught.value.code == FAIL_PROFILE_IO


def test_bound_source_drift_fails_closed(tmp_path: Path) -> None:
    copied = _copy_bound_sources(tmp_path)
    attacked_path = copied["runtime_seccomp"]
    attacked_path.write_bytes(attacked_path.read_bytes() + b"\n")

    with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
        load_phase2b_c1_container_actor_profile_v1(
            PROFILE_PATH,
            repository_root=tmp_path,
        )
    assert caught.value.code == FAIL_PROFILE_BINDING


def test_hardlinked_and_oversized_profile_files_are_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "profile.json"
    shutil.copyfile(PROFILE_PATH, copied)
    alias = tmp_path / "profile-alias.json"
    os.link(copied, alias)
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as hardlink:
        load_phase2b_c1_container_actor_profile_v1(copied)
    assert hardlink.value.code == FAIL_PROFILE_IO

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * (256 * 1024))
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as too_large:
        load_phase2b_c1_container_actor_profile_v1(oversized)
    assert too_large.value.code == FAIL_PROFILE_IO


def test_profile_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "profile.json"
    os.mkfifo(fifo, 0o600)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_fifo_load_worker,
        args=(str(fifo), str(profile_module.PROJECT_ROOT), output),
    )
    process.start()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        pytest.fail("profile FIFO read blocked")
    assert process.exitcode == 0
    assert output.get(timeout=1) == FAIL_PROFILE_IO


@pytest.mark.parametrize("attack", ("symlink", "hardlink", "fifo", "oversize"))
def test_bound_source_namespace_attacks_fail_closed(tmp_path: Path, attack: str) -> None:
    copied = _copy_bound_sources(tmp_path)
    target = copied["runtime_seccomp"]
    original = target.read_bytes()
    target.unlink()
    if attack == "symlink":
        target.symlink_to(profile_module.PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json")
    elif attack == "hardlink":
        backing = tmp_path / "backing.json"
        backing.write_bytes(original)
        os.link(backing, target)
    elif attack == "fifo":
        os.mkfifo(target, 0o600)
    else:
        target.write_bytes(b"x" * (16 * 1024 * 1024 + 1))

    if attack == "fifo":
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        process = context.Process(
            target=_fifo_load_worker,
            args=(str(PROFILE_PATH), str(tmp_path), output),
        )
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            pytest.fail("bound-source FIFO read blocked")
        assert process.exitcode == 0
        assert output.get(timeout=1) == FAIL_PROFILE_BINDING
    else:
        with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
            load_phase2b_c1_container_actor_profile_v1(
                PROFILE_PATH,
                repository_root=tmp_path,
            )
        assert caught.value.code == FAIL_PROFILE_BINDING


def test_bound_source_namespace_swap_during_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = _copy_bound_sources(tmp_path)
    target = copied["runtime_seccomp"]
    original_read = profile_module.os.read
    swapped = False

    def attacking_read(descriptor: int, maximum: int) -> bytes:
        nonlocal swapped
        try:
            opened = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            opened = Path("/")
        if not swapped and opened == target:
            swapped = True
            target.unlink()
            target.symlink_to(PROFILE_PATH)
        return original_read(descriptor, maximum)

    monkeypatch.setattr(profile_module.os, "read", attacking_read)
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
        load_phase2b_c1_container_actor_profile_v1(
            PROFILE_PATH,
            repository_root=tmp_path,
        )
    assert swapped is True
    assert caught.value.code == FAIL_PROFILE_BINDING


def test_wrong_runtime_argument_types_fail_before_any_external_action() -> None:
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV1) as caught:
        load_phase2b_c1_container_actor_profile_v1(str(PROFILE_PATH))
    assert caught.value.code == FAIL_PROFILE_IO
