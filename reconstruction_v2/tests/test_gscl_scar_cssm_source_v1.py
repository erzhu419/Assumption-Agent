from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

import pytest

from assumption_agent.benchmarks import gscl_scar_cssm_source_v1 as scar


_SECRET = bytes(range(scar.HMAC_SECRET_BYTES))
_STUDY_ID = "SCAR_CSSM_INTRINSIC_V1"


def _official_source() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "gscl_intrinsic_candidates_20260730"
        / "repos"
        / "scar"
        / "release"
        / "system_analogy_en.json"
    )


def _private_official_copy(tmp_path: Path) -> Path:
    source = _official_source()
    if not source.is_file():
        pytest.skip("ignored local SCAR-English source is unavailable")
    destination = tmp_path / "system_analogy_en.private.jsonl"
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return destination


@pytest.fixture(scope="module")
def official_compilation(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("scar_cssm_official")
    source = _private_official_copy(root)
    before = (_official_source().stat().st_mode, _official_source().read_bytes())
    result = scar.compile_scar_cssm_source_v1(
        source, secret=_SECRET, study_id=_STUDY_ID
    )
    after = (_official_source().stat().st_mode, _official_source().read_bytes())
    assert before == after
    return result


def _synthetic_row(
    source_id: int,
    *,
    mappings: list[list[str]] | None = None,
    explanation: Any = None,
) -> dict[str, Any]:
    if mappings is None:
        mappings = [["alpha", "one"], ["beta", "two"]]
    if explanation is None:
        explanation = ["deliberately ignored"]
    return {
        "id": source_id,
        "lang": "en",
        "system_a": f"system a {source_id}",
        "system_b": f"system b {source_id}",
        "mappings": mappings,
        "system_a_domain": "DomainA",
        "system_b_domain": "DomainB",
        "system_a_background": "Alpha changes beta.",
        "system_b_background": "One changes two.",
        "Explanation": explanation,
    }


def _jsonl(*rows: dict[str, Any]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _parse_synthetic(*rows: dict[str, Any]):
    return scar._parse_source_rows(  # noqa: SLF001
        _jsonl(*rows),
        expected_row_count=len(rows),
        expected_mapping_count=sum(len(row["mappings"]) for row in rows),
        expected_ids=frozenset(range(1, len(rows) + 1)),
    )


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _recommit_action_pack(action_pack: dict[str, Any]) -> None:
    final_keys = {
        "action_commitment_sha256",
        "cross_binding_hmac_sha256",
        "label_commitment_sha256",
        "self_sha256",
    }
    action_core = {
        key: value for key, value in action_pack.items() if key not in final_keys
    }
    action_pack["action_commitment_sha256"] = scar._content_hash(  # noqa: SLF001
        action_core
    )
    action_body = {
        key: value for key, value in action_pack.items() if key != "self_sha256"
    }
    action_pack["self_sha256"] = scar._content_hash(action_body)  # noqa: SLF001


def test_frozen_source_and_cohort_commitments_are_exact() -> None:
    assert scar.EXPECTED_SOURCE_SIZE_BYTES == 1_393_355
    assert scar.EXPECTED_SOURCE_SHA256 == (
        "12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369"
    )
    assert scar.EXPECTED_SOURCE_ROW_COUNT == 400
    assert scar.EXPECTED_SOURCE_MAPPING_COUNT == 1_618
    assert set(scar.PUBLIC_DENY_RAW_LINE_SHA256_BY_ID) == {
        12,
        29,
        55,
        85,
        107,
        129,
        180,
        347,
        351,
    }
    assert 347 in scar.PUBLIC_DENY_RAW_LINE_SHA256_BY_ID
    assert 90 not in scar.PUBLIC_DENY_RAW_LINE_SHA256_BY_ID
    assert all(
        len(value) == 64
        for value in scar.PUBLIC_DENY_RAW_LINE_SHA256_BY_ID.values()
    )
    assert scar.EXPECTED_PUBLIC_DENY_LINE_HASH_LIST_SHA256 == (
        "08ed304f8ba4033d0e84e7b0f13a14557d9757b90e5b1687098940c32369eff9"
    )
    assert len(scar.EXPECTED_NORMALIZED_DUPLICATE_SLOT_ROW_IDS) == 29
    assert scar.EXPECTED_AMBIGUOUS_LINE_HASH_LIST_SHA256 == (
        "3067b68aedb13f41006717149a3f6b77418ea9c79ad4d5ba2ccc493225917fe5"
    )
    assert scar.EXPECTED_PRIMARY_ROW_COUNT == 362
    assert scar.EXPECTED_PRIMARY_MAPPING_COUNT == 1_339
    assert scar.EXPECTED_PRIMARY_LINE_HASH_LIST_SHA256 == (
        "c9ed6bc9967b9fe0bb5373868b2363c3a3b2f293c84e1ad53ec58b0a28822916"
    )
    assert scar.VARIANT_NAMES == ("base", "system_swap")


def test_strict_synthetic_parser_ignores_explanation_contents() -> None:
    rows = _parse_synthetic(
        _synthetic_row(
            1,
            explanation=[None, {"not": "consumed"}, [1, 2, 3]],
        )
    )
    assert len(rows) == 1
    assert len(rows[0].mappings) == 2
    assert not hasattr(rows[0], "explanation")


@pytest.mark.parametrize(
    ("mutator", "issue_id"),
    [
        (
            lambda row: row.update({"lang": "zh"}),
            "SCAR_SOURCE_ID_OR_LANG_INVALID",
        ),
        (
            lambda row: row.update({"id": True}),
            "SCAR_SOURCE_ID_OR_LANG_INVALID",
        ),
        (
            lambda row: row.update({"system_a": ""}),
            "SCAR_SOURCE_TEXT_FIELD_INVALID",
        ),
        (
            lambda row: row.update({"mappings": "not-a-list"}),
            "SCAR_SOURCE_SEQUENCE_FIELD_INVALID",
        ),
        (
            lambda row: row.update({"Explanation": "not-a-list"}),
            "SCAR_SOURCE_SEQUENCE_FIELD_INVALID",
        ),
        (
            lambda row: row.update({"mappings": [["alpha"]]}),
            "SCAR_SOURCE_MAPPING_INVALID",
        ),
        (
            lambda row: row.update({"mappings": [["alpha", "one"]]}),
            "SCAR_SOURCE_ARITY_INVALID",
        ),
        (
            lambda row: row.update(
                {
                    "mappings": [
                        ["alpha", "one"],
                        ["alpha", "one"],
                    ]
                }
            ),
            "SCAR_SOURCE_MAPPING_DUPLICATE",
        ),
        (
            lambda row: row.update({"unexpected": "field"}),
            "SCAR_SOURCE_SCHEMA_INVALID",
        ),
    ],
)
def test_strict_synthetic_parser_rejects_bad_shapes(mutator, issue_id) -> None:
    row = _synthetic_row(1)
    mutator(row)
    mapping_count = len(row["mappings"]) if isinstance(row["mappings"], list) else 0
    with pytest.raises(scar.ScarCssmSourceError) as caught:
        scar._parse_source_rows(  # noqa: SLF001
            _jsonl(row),
            expected_row_count=1,
            expected_mapping_count=mapping_count,
            expected_ids=frozenset({1}),
        )
    assert caught.value.issue_id == issue_id


def test_strict_synthetic_parser_rejects_duplicate_json_keys() -> None:
    row = _jsonl(_synthetic_row(1)).rstrip(b"\n")
    duplicated = row[:-1] + b',"lang":"en"}\n'
    with pytest.raises(scar.ScarCssmSourceError) as caught:
        scar._parse_source_rows(  # noqa: SLF001
            duplicated,
            expected_row_count=1,
            expected_mapping_count=2,
            expected_ids=frozenset({1}),
        )
    assert caught.value.issue_id == "SCAR_SOURCE_JSON_INVALID"


def test_normalized_duplicate_detection_is_side_local() -> None:
    rows = _parse_synthetic(
        _synthetic_row(
            1,
            mappings=[["Ａ  B", "one"], ["a b", "two"]],
        ),
        _synthetic_row(
            2,
            mappings=[["alpha", "same"], ["beta", " SAME "]],
        ),
    )
    assert all(scar._has_normalized_duplicate_slot(row) for row in rows)  # noqa: SLF001


def test_hmac_item_and_side_orders_are_deterministic_and_domain_separated() -> None:
    rows = _parse_synthetic(_synthetic_row(1))
    first_action, first_label = scar._build_core_packs(  # noqa: SLF001
        rows, secret=_SECRET, study_id=_STUDY_ID
    )
    replay_action, replay_label = scar._build_core_packs(  # noqa: SLF001
        rows, secret=_SECRET, study_id=_STUDY_ID
    )
    other_action, _ = scar._build_core_packs(  # noqa: SLF001
        rows, secret=b"z" * scar.HMAC_SECRET_BYTES, study_id=_STUDY_ID
    )
    assert (first_action, first_label) == (replay_action, replay_label)
    assert first_action["items"][0]["item_token"] != (
        other_action["items"][0]["item_token"]
    )
    token = first_action["items"][0]["item_token"]
    slot = first_action["items"][0]["variants"]["base"]["left"]["slots"][0][
        "opaque_slot_id"
    ]
    assert scar._slot_order_key(  # noqa: SLF001
        _SECRET, _STUDY_ID, token, "a", slot
    ) != scar._slot_order_key(  # noqa: SLF001
        _SECRET, _STUDY_ID, token, "b", slot
    )


def test_official_copy_qualifies_with_exact_safe_aggregate(
    official_compilation,
) -> None:
    result = official_compilation
    scar.validate_scar_cssm_pack_binding_v1(
        result.action_pack,
        result.label_pack,
        secret=_SECRET,
        study_id=_STUDY_ID,
    )
    safe = result.safe_aggregate
    assert safe["status"] == "qualified"
    assert safe["source_binding"] == {
        "row_count": 400,
        "sha256": scar.EXPECTED_SOURCE_SHA256,
        "size_bytes": scar.EXPECTED_SOURCE_SIZE_BYTES,
        "total_mapping_count": 1_618,
    }
    assert safe["access_counts"] == {
        "model_call_count": 0,
        "network_call_count": 0,
        "scorer_call_count": 0,
        "source_access_count": 1,
    }
    assert safe["action_item_count"] == 391
    assert safe["action_variant_count"] == 782
    assert safe["primary_row_count"] == 362
    assert safe["primary_mapping_count"] == 1_339
    assert safe["ambiguous_row_count"] == 29
    assert safe["ambiguous_mapping_count"] == 233
    body = {key: value for key, value in safe.items() if key != "self_sha256"}
    assert safe["self_sha256"] == scar._content_hash(body)  # noqa: SLF001


def test_action_only_worker_validator_reads_no_source_label_or_secret(
    official_compilation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("action-only validator attempted file I/O")

    def forbidden_cross_binding(*_args, **_kwargs):
        raise AssertionError("action-only validator attempted label validation")

    monkeypatch.setattr(scar.os, "open", forbidden_open)
    monkeypatch.setattr(
        scar,
        "validate_scar_cssm_pack_binding_v1",
        forbidden_cross_binding,
    )
    scar.validate_scar_cssm_action_pack_v1(
        official_compilation.action_pack,
        study_id=_STUDY_ID,
    )


def test_action_only_worker_validator_rejects_tamper_and_wrong_study(
    official_compilation,
) -> None:
    tampered = copy.deepcopy(official_compilation.action_pack)
    tampered["items"][0]["variants"]["base"]["left"]["system"] += " tamper"
    with pytest.raises(scar.ScarCssmSourceError) as content_error:
        scar.validate_scar_cssm_action_pack_v1(
            tampered,
            study_id=_STUDY_ID,
        )
    assert content_error.value.issue_id == "SCAR_PACK_COMMITMENT_INVALID"

    with pytest.raises(scar.ScarCssmSourceError) as study_error:
        scar.validate_scar_cssm_action_pack_v1(
            official_compilation.action_pack,
            study_id="OTHER_STUDY",
        )
    assert study_error.value.issue_id == "SCAR_STUDY_CROSS_BINDING_INVALID"


def test_action_only_worker_rejects_recommitted_forbidden_key_and_topology(
    official_compilation,
) -> None:
    leaked = copy.deepcopy(official_compilation.action_pack)
    leaked["items"][0]["variants"]["base"]["left"]["mappings"] = []
    _recommit_action_pack(leaked)
    with pytest.raises(scar.ScarCssmSourceError) as leakage_error:
        scar.validate_scar_cssm_action_pack_v1(leaked, study_id=_STUDY_ID)
    assert leakage_error.value.issue_id == "SCAR_ACTION_LABEL_LEAKAGE"

    short = copy.deepcopy(official_compilation.action_pack)
    short["items"].pop()
    _recommit_action_pack(short)
    with pytest.raises(scar.ScarCssmSourceError) as topology_error:
        scar.validate_scar_cssm_action_pack_v1(short, study_id=_STUDY_ID)
    assert topology_error.value.issue_id == "SCAR_ACTION_PACK_INVALID"


def test_action_only_worker_scopes_label_and_cross_fields_to_hex_presence(
    official_compilation,
) -> None:
    action = copy.deepcopy(official_compilation.action_pack)
    action["label_commitment_sha256"] = "0" * 64
    action["cross_binding_hmac_sha256"] = "f" * 64
    action_body = {
        key: value for key, value in action.items() if key != "self_sha256"
    }
    action["self_sha256"] = scar._content_hash(action_body)  # noqa: SLF001
    scar.validate_scar_cssm_action_pack_v1(action, study_id=_STUDY_ID)

    malformed = copy.deepcopy(action)
    malformed["label_commitment_sha256"] = "not-a-sha256"
    malformed_body = {
        key: value for key, value in malformed.items() if key != "self_sha256"
    }
    malformed["self_sha256"] = scar._content_hash(  # noqa: SLF001
        malformed_body
    )
    with pytest.raises(scar.ScarCssmSourceError) as caught:
        scar.validate_scar_cssm_action_pack_v1(
            malformed,
            study_id=_STUDY_ID,
        )
    assert caught.value.issue_id == "SCAR_PACK_COMMITMENT_INVALID"


def test_action_pack_is_recursively_label_free_and_source_opaque(
    official_compilation,
) -> None:
    action = official_compilation.action_pack
    forbidden = {
        "Explanation",
        "cohort",
        "domain",
        "explanation",
        "gold_pairs",
        "id",
        "index",
        "mapping",
        "mapping_index",
        "mappings",
        "original_index",
        "raw_id",
        "source_id",
        "strata",
        "system_a_domain",
        "system_b_domain",
    }
    assert forbidden.isdisjoint(_walk_keys(action["items"]))
    assert len(action["items"]) == 391
    assert all(
        set(item) == {"item_token", "variants"} for item in action["items"]
    )
    assert all(
        item["item_token"].startswith("scar-item-v1-")
        for item in action["items"]
    )


def test_base_and_system_swap_are_the_only_exact_variants(
    official_compilation,
) -> None:
    all_slots: set[str] = set()
    for item in official_compilation.action_pack["items"]:
        variants = item["variants"]
        assert tuple(variants) == ("base", "system_swap")
        assert variants["system_swap"]["left"] == variants["base"]["right"]
        assert variants["system_swap"]["right"] == variants["base"]["left"]
        left_ids = {
            row["opaque_slot_id"]
            for row in variants["base"]["left"]["slots"]
        }
        right_ids = {
            row["opaque_slot_id"]
            for row in variants["base"]["right"]["slots"]
        }
        assert len(left_ids) == len(right_ids)
        assert left_ids.isdisjoint(right_ids)
        assert all_slots.isdisjoint(left_ids | right_ids)
        all_slots.update(left_ids | right_ids)


def test_gold_pairs_cover_opaque_slots_and_swap_only_reverses(
    official_compilation,
) -> None:
    actions = {
        item["item_token"]: item
        for item in official_compilation.action_pack["items"]
    }
    primary = ambiguous = primary_pairs = ambiguous_pairs = 0
    for label in official_compilation.label_pack["items"]:
        action = actions[label["item_token"]]
        left = {
            row["opaque_slot_id"]
            for row in action["variants"]["base"]["left"]["slots"]
        }
        right = {
            row["opaque_slot_id"]
            for row in action["variants"]["base"]["right"]["slots"]
        }
        base = label["gold_pairs"]["base"]
        swapped = label["gold_pairs"]["system_swap"]
        assert {pair[0] for pair in base} == left
        assert {pair[1] for pair in base} == right
        assert swapped == [[target, source] for source, target in base]
        if label["strata"]["cohort"] == "primary_unique_slot":
            primary += 1
            primary_pairs += len(base)
        else:
            ambiguous += 1
            ambiguous_pairs += len(base)
    assert (primary, primary_pairs) == (362, 1_339)
    assert (ambiguous, ambiguous_pairs) == (29, 233)


@pytest.mark.parametrize("target", ["action", "label"])
def test_pack_tamper_fails_cross_binding(official_compilation, target) -> None:
    action = copy.deepcopy(official_compilation.action_pack)
    label = copy.deepcopy(official_compilation.label_pack)
    if target == "action":
        action["items"][0]["variants"]["base"]["left"]["system"] += " tamper"
    else:
        label["items"][0]["strata"]["arity"] += 1
    with pytest.raises(scar.ScarCssmSourceError) as caught:
        scar.validate_scar_cssm_pack_binding_v1(
            action,
            label,
            secret=_SECRET,
            study_id=_STUDY_ID,
        )
    assert caught.value.issue_id in {
        "SCAR_PACK_COMMITMENT_INVALID",
        "SCAR_PACK_SELF_HASH_INVALID",
    }


def test_wrong_secret_or_study_cannot_cross_bind(official_compilation) -> None:
    with pytest.raises(scar.ScarCssmSourceError) as wrong_secret:
        scar.validate_scar_cssm_pack_binding_v1(
            official_compilation.action_pack,
            official_compilation.label_pack,
            secret=b"x" * scar.HMAC_SECRET_BYTES,
            study_id=_STUDY_ID,
        )
    assert wrong_secret.value.issue_id == "SCAR_PACK_COMMITMENT_INVALID"
    with pytest.raises(scar.ScarCssmSourceError) as wrong_study:
        scar.validate_scar_cssm_pack_binding_v1(
            official_compilation.action_pack,
            official_compilation.label_pack,
            secret=_SECRET,
            study_id="OTHER_STUDY",
        )
    assert wrong_study.value.issue_id == "SCAR_STUDY_CROSS_BINDING_INVALID"


def test_safe_aggregate_contains_no_item_or_content_surface(
    official_compilation,
) -> None:
    encoded = json.dumps(
        official_compilation.safe_aggregate,
        ensure_ascii=True,
        sort_keys=True,
    )
    for forbidden in (
        "scar-item-v1-",
        "scar-slot-v1-",
        '"items"',
        '"gold_pairs"',
        '"background"',
        '"surface"',
        '"system_a"',
        '"system_b"',
        '"Explanation"',
    ):
        assert forbidden not in encoded


def test_source_is_opened_once_and_all_later_work_is_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _private_official_copy(tmp_path)
    real_open = scar.os.open
    opened: list[Path] = []

    def counted_open(path, flags, *args):
        if Path(path) == source:
            opened.append(Path(path))
        return real_open(path, flags, *args)

    monkeypatch.setattr(scar.os, "open", counted_open)
    scar.compile_scar_cssm_source_v1(
        source,
        secret=_SECRET,
        study_id="SCAR_CSSM_SINGLE_OPEN_TEST",
    )
    assert opened == [source]


def test_source_requires_non_symlink_regular_mode_0600(tmp_path: Path) -> None:
    source = _official_source()
    if not source.is_file():
        pytest.skip("ignored local SCAR-English source is unavailable")
    public = tmp_path / "public.jsonl"
    shutil.copyfile(source, public)
    public.chmod(0o644)
    with pytest.raises(scar.ScarCssmSourceError) as wrong_mode:
        scar.compile_scar_cssm_source_v1(
            public, secret=_SECRET, study_id="SCAR_BAD_MODE"
        )
    assert wrong_mode.value.issue_id == "SCAR_SOURCE_FILE_CONTRACT_INVALID"

    public.chmod(0o600)
    link = tmp_path / "source-link.jsonl"
    link.symlink_to(public)
    with pytest.raises(scar.ScarCssmSourceError) as symlink:
        scar.compile_scar_cssm_source_v1(
            link, secret=_SECRET, study_id="SCAR_BAD_LINK"
        )
    assert symlink.value.issue_id == "SCAR_SOURCE_FILE_CONTRACT_INVALID"


def test_same_size_source_tamper_fails_exact_sha(tmp_path: Path) -> None:
    source = _private_official_copy(tmp_path)
    descriptor = os.open(source, os.O_RDWR)
    try:
        first = os.read(descriptor, 1)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, b"[" if first != b"[" else b"{")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert source.stat().st_size == scar.EXPECTED_SOURCE_SIZE_BYTES
    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    with pytest.raises(scar.ScarCssmSourceError) as caught:
        scar.compile_scar_cssm_source_v1(
            source, secret=_SECRET, study_id="SCAR_BAD_SHA"
        )
    assert caught.value.issue_id == "SCAR_SOURCE_IDENTITY_INVALID"


def test_secret_and_study_contracts_fail_before_source_access(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    with pytest.raises(scar.ScarCssmSourceError) as bad_secret:
        scar.compile_scar_cssm_source_v1(
            absent, secret=b"short", study_id=_STUDY_ID
        )
    assert bad_secret.value.issue_id == "SCAR_HMAC_SECRET_INVALID"
    with pytest.raises(scar.ScarCssmSourceError) as bad_study:
        scar.compile_scar_cssm_source_v1(
            absent, secret=_SECRET, study_id="bad study id"
        )
    assert bad_study.value.issue_id == "SCAR_STUDY_ID_INVALID"


def test_official_source_hash_is_unchanged_after_tests() -> None:
    source = _official_source()
    if not source.is_file():
        pytest.skip("ignored local SCAR-English source is unavailable")
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == scar.EXPECTED_SOURCE_SHA256
    )
