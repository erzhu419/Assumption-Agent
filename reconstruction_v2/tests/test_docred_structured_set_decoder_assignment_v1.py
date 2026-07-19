from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Callable

import pytest

from assumption_agent.benchmarks import (
    docred_structured_set_decoder_assignment_v1 as assignment,
)
from assumption_agent.benchmarks import (
    docred_structured_set_decoder_source_qualification_v1 as qualifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_SECRET = bytes(range(32))
PRIVATE_MARKERS = (
    "PRIVATE_TITLE",
    "PRIVATE_HEAD",
    "PRIVATE_TAIL",
    "PRIVATE_SENTENCE",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _with_self_hash(value: dict[str, Any], field: str) -> bytes:
    body = copy.deepcopy(value)
    body.pop(field, None)
    body[field] = hashlib.sha256(_canonical(body)).hexdigest()
    return json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("ascii") + b"\n"


def _write_0600(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _document(split: str, family: str, index: int) -> dict[str, Any]:
    marker = f"PRIVATE_{split}_{family}_{index}"
    relation = qualifier.FAMILY_PROPERTIES[family][0]
    head = f"PRIVATE_HEAD_{marker}"
    tail = f"PRIVATE_TAIL_{marker}"
    sentences = [
        [head, "PRIVATE_SENTENCE", "zero"],
        [tail, "PRIVATE_SENTENCE", "one"],
    ]
    sentences.extend(
        [[f"PRIVATE_SENTENCE_{marker}_{ordinal}"] for ordinal in range(2, 10)]
    )
    return {
        "title": f"PRIVATE_TITLE_{marker}",
        "sents": sentences,
        "vertexSet": [
            [
                {
                    "name": head,
                    "sent_id": 0,
                    "pos": [0, 1],
                    "type": "PER",
                },
                {
                    "name": head,
                    "sent_id": 0,
                    "pos": [0, 1],
                    "type": "PER",
                },
            ],
            [
                {
                    "name": tail,
                    "sent_id": 1,
                    "pos": [0, 1],
                    "type": "ORG",
                }
            ],
        ],
        "labels": [{"h": 0, "t": 1, "r": relation, "evidence": [0, 1]}],
    }


def _documents(split: str, per_family: int) -> list[dict[str, Any]]:
    return [
        _document(split, family, index)
        for family in qualifier.FAMILIES
        for index in range(per_family)
    ]


def _rebind_source_chain(project: Path) -> None:
    manifests = project / "manifests"
    source_root = project / "artifacts/docred_official_source_v1"
    sources = {
        "train_annotated": (
            source_root / "train_annotated.json",
            "train_sha256",
        ),
        "dev": (source_root / "dev.json", "dev_sha256"),
        "relation_metadata": (
            source_root / "rel_info.json",
            "relation_metadata_sha256",
        ),
    }
    access_path = manifests / (
        "docred_structured_set_decoder_source_access_v1.json"
    )
    access = json.loads(access_path.read_text("utf-8"))
    for key, (path, _prerequisite_key) in sources.items():
        raw = path.read_bytes()
        access["allowed_local_files"][key].update(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "mode": 0o600,
            }
        )
    access_path.write_bytes(_with_self_hash(access, "source_access_sha256"))
    access = json.loads(access_path.read_text("utf-8"))

    k3_path = manifests / (
        "docred_structured_set_decoder_k3_pre_row_amendment_v1.json"
    )
    k3 = json.loads(k3_path.read_text("utf-8"))
    k3["prerequisite"]["source_access_self_sha256"] = access[
        "source_access_sha256"
    ]
    for key, (_path, prerequisite_key) in sources.items():
        k3["prerequisite"][prerequisite_key] = access["allowed_local_files"][
            key
        ]["sha256"]
    k3_path.write_bytes(_with_self_hash(k3, "k3_amendment_sha256"))
    k3 = json.loads(k3_path.read_text("utf-8"))

    family_path = manifests / (
        "docred_structured_set_decoder_relation_family_freeze_v1.json"
    )
    family = json.loads(family_path.read_text("utf-8"))
    family["prerequisite"]["source_access_self_sha256"] = access[
        "source_access_sha256"
    ]
    family["prerequisite"]["k_amendment_self_sha256"] = k3[
        "k3_amendment_sha256"
    ]
    for key, (_path, prerequisite_key) in sources.items():
        family["prerequisite"][prerequisite_key] = access[
            "allowed_local_files"
        ][key]["sha256"]
    family_path.write_bytes(
        _with_self_hash(family, "relation_family_freeze_sha256")
    )


def _fixture(
    tmp_path: Path,
    *,
    suffix: str,
    train_per_family: int = 60,
    dev_per_family: int = 20,
) -> tuple[Path, Path]:
    fixture_id = hashlib.sha256(
        (str(tmp_path) + "\x00" + suffix).encode("utf-8")
    ).hexdigest()[:20]
    base = Path("/tmp/docred_assignment_v1_tests") / fixture_id
    shutil.rmtree(base, ignore_errors=True)
    project = base / "project"
    output = base / "output"
    manifests = project / "manifests"
    manifests.mkdir(parents=True)
    output.mkdir(parents=True)
    for name in (
        "docred_structured_set_decoder_source_custody_v1.json",
        "docred_structured_set_decoder_source_access_v1.json",
        "docred_structured_set_decoder_k3_pre_row_amendment_v1.json",
        "docred_structured_set_decoder_relation_family_freeze_v1.json",
    ):
        (manifests / name).write_bytes(
            (PROJECT_ROOT / "manifests" / name).read_bytes()
        )
    relation_metadata = {
        property_id: f"frozen relation description number {ordinal}"
        for ordinal, property_id in enumerate(
            sorted(qualifier.FAMILY_PROPERTY_UNION)
        )
    }
    source_root = project / "artifacts/docred_official_source_v1"
    _write_0600(
        source_root / "train_annotated.json",
        _json_bytes(_documents("train", train_per_family)),
    )
    _write_0600(
        source_root / "dev.json",
        _json_bytes(_documents("dev", dev_per_family)),
    )
    _write_0600(source_root / "rel_info.json", _json_bytes(relation_metadata))
    _write_0600(source_root / "test.json", b"PRIVATE_TEST_DO_NOT_OPEN")
    _write_0600(
        source_root / "train_distant.json", b"PRIVATE_DISTANT_DO_NOT_OPEN"
    )
    _rebind_source_chain(project)
    return project, output


def _contains_list(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_contains_list(child) for child in value.values())
    return False


def _verify_self_hash(value: dict[str, Any], field: str) -> None:
    expected = value[field]
    body = dict(value)
    del body[field]
    assert hashlib.sha256(_canonical(body)).hexdigest() == expected


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_hall_deficiency_and_component_cap_are_detected() -> None:
    target_a = ("A", "family")
    target_b = ("B", "family")
    shared = {
        component: {
            target_a: assignment._EdgeChoice(component + 1, (), f"{component}-A"),
            target_b: assignment._EdgeChoice(component + 1, (), f"{component}-B"),
        }
        for component in range(3)
    }
    hall = assignment._deterministic_min_cost_assignment(
        shared,
        {target_a: 2, target_b: 2},
    )
    assert hall.assigned_count == 3
    assert hall.required_count == 4

    one_component = assignment._deterministic_min_cost_assignment(
        {0: shared[0]},
        {target_a: 1, target_b: 1},
    )
    assert one_component.assigned_count == 1
    assert sum(len(values) for values in one_component.selected.values()) == 1


def test_min_cost_flow_avoids_greedy_false_shortfall() -> None:
    target_a = ("A", "family")
    target_b = ("B", "family")
    solution = assignment._deterministic_min_cost_assignment(
        {
            0: {
                target_a: assignment._EdgeChoice(0, (), "flex-A"),
                target_b: assignment._EdgeChoice(0, (), "flex-B"),
            },
            1: {
                target_a: assignment._EdgeChoice(1, (), "only-A"),
            },
        },
        {target_a: 1, target_b: 1},
    )
    assert solution.assigned_count == 2
    assert solution.selected[target_a] == ("only-A",)
    assert solution.selected[target_b] == ("flex-B",)


def test_success_commits_exact_private_contract_and_opens_sources_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, output = _fixture(tmp_path, suffix="success")
    source_opens: list[str] = []
    private_path_opens: list[str] = []
    original_open = Path.open

    def spy_open(path: Path, *args: Any, **kwargs: Any):
        if path.name in {
            "train_annotated.json",
            "dev.json",
            "rel_info.json",
            "test.json",
            "train_distant.json",
        }:
            assert (output / assignment.ATTEMPT_MARKER_NAME).is_file()
            source_opens.append(path.name)
        if assignment.PRIVATE_DIRECTORY_NAME in path.parts:
            private_path_opens.append(path.as_posix())
        return original_open(path, *args, **kwargs)

    secret_calls: list[int] = []

    def secret_factory(size: int) -> bytes:
        secret_calls.append(size)
        return FIXED_SECRET

    monkeypatch.setattr(Path, "open", spy_open)
    receipt = assignment.run_synthetic_one_shot(
        project,
        output,
        secret_factory=secret_factory,
    )
    assert receipt["status"] == "passed_assignment_committed_no_private_values"
    assert secret_calls == [32]
    assert sorted(source_opens) == ["dev.json", "rel_info.json", "train_annotated.json"]
    assert private_path_opens == []
    assert not _contains_list(receipt)
    public_raw = _canonical(receipt).decode("ascii")
    assert not any(marker in public_raw for marker in PRIVATE_MARKERS)
    assert "i_" not in public_raw
    _verify_self_hash(receipt, "assignment_receipt_sha256")
    qualification_path = output / assignment.SOURCE_QUALIFICATION_RECEIPT_NAME
    assert qualification_path.is_file()
    qualification_raw = qualification_path.read_bytes()
    qualification_receipt = json.loads(qualification_raw)
    assert qualification_receipt["status"] == (
        "passed_source_qualification_no_selection"
    )
    assert receipt["qualification_receipt_file_sha256"] == hashlib.sha256(
        qualification_raw
    ).hexdigest()

    private_root = output / assignment.PRIVATE_DIRECTORY_NAME
    secret_path = private_root / "selection_secret.bin"
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
    assert secret_path.read_bytes() == FIXED_SECRET
    assert receipt["assignment_aggregate"][
        "selection_secret_commitment_sha256"
    ] == assignment._secret_commitment(FIXED_SECRET)
    all_ids: set[str] = set()
    for block, _split, quota, has_labels in assignment.BLOCK_SPECS:
        block_root = private_root / block
        view_path = block_root / "view.json"
        assert stat.S_IMODE(view_path.stat().st_mode) == 0o600
        view = json.loads(view_path.read_text("ascii"))
        assert len(view["items"]) == quota * len(qualifier.FAMILIES)
        for item in view["items"]:
            assert set(item) == {"item_id", "query", "corpus", "agent_sidecar"}
            serialized = _canonical(item).decode("ascii")
            assert re.search(r"\bP[1-9][0-9]*\b", serialized) is None
            assert all(
                forbidden not in item
                for forbidden in ("family", "block", "title", "gold")
            )
            assert item["item_id"] not in all_ids
            all_ids.add(item["item_id"])
            assert item["query"].startswith("HEAD: ")
            assert "\nRELATION: " in item["query"]
            assert "\nTAIL: " in item["query"]
            assert len(item["corpus"]) == 10
            assert set(item["agent_sidecar"]) == {
                "head_entity_index",
                "tail_entity_index",
                "entities",
            }
        labels_path = block_root / "labels.json"
        assert labels_path.exists() is has_labels
        if has_labels:
            assert stat.S_IMODE(labels_path.stat().st_mode) == 0o600
            labels = json.loads(labels_path.read_text("ascii"))
            assert len(labels["items"]) == len(view["items"])
            assert {item["item_id"] for item in labels["items"]} == {
                item["item_id"] for item in view["items"]
            }
            assert all(
                set(item) == {"item_id", "gold_sentence_ordinals"}
                for item in labels["items"]
            )
    assert len(all_ids) == assignment.TOTAL_REQUIRED_ITEMS == 240
    assert not (private_root / "F_search" / "labels.json").exists()
    assert (private_root / "M_search" / "view.json").is_file()
    assert (private_root / "M_search" / "labels.json").is_file()
    for block in assignment.BLOCK_ORDER:
        aggregate = receipt["assignment_aggregate"]["block_aggregates"][block]
        assert aggregate["view_item_count"] == (
            assignment.BLOCK_TO_QUOTA[block] * len(qualifier.FAMILIES)
        )
        assert aggregate["family_item_counts"] == {
            family: assignment.BLOCK_TO_QUOTA[block]
            for family in qualifier.FAMILIES
        }


def test_fixed_secret_is_byte_deterministic(tmp_path: Path) -> None:
    project_a, output_a = _fixture(tmp_path, suffix="deterministic-a")
    project_b, output_b = _fixture(tmp_path, suffix="deterministic-b")
    receipt_a = assignment.run_synthetic_one_shot(
        project_a, output_a, secret_factory=lambda size: FIXED_SECRET
    )
    receipt_b = assignment.run_synthetic_one_shot(
        project_b, output_b, secret_factory=lambda size: FIXED_SECRET
    )
    assert receipt_a == receipt_b
    assert _tree_bytes(output_a / assignment.PRIVATE_DIRECTORY_NAME) == (
        _tree_bytes(output_b / assignment.PRIVATE_DIRECTORY_NAME)
    )


def test_qualification_shortfall_never_requests_secret(tmp_path: Path) -> None:
    project, output = _fixture(
        tmp_path,
        suffix="shortfall",
        train_per_family=59,
    )
    calls: list[int] = []

    def forbidden_secret(size: int) -> bytes:
        calls.append(size)
        return FIXED_SECRET

    incident = assignment.run_synthetic_one_shot(
        project,
        output,
        secret_factory=forbidden_secret,
    )
    assert incident["status"] == "terminal_no_replay_no_private_values"
    assert incident["failure_category"] == "source_qualification_terminal"
    assert calls == []
    assert not (output / assignment.PRIVATE_DIRECTORY_NAME).exists()
    assert (output / assignment.TERMINAL_INCIDENT_NAME).is_file()
    assert (output / assignment.SOURCE_QUALIFICATION_RECEIPT_NAME).is_file()
    _verify_self_hash(incident, "terminal_incident_sha256")


@pytest.mark.parametrize("kind", ("marker", "private_output"))
def test_preexisting_marker_or_output_is_refused_without_replay(
    tmp_path: Path,
    kind: str,
) -> None:
    project, output = _fixture(tmp_path, suffix="preexisting-" + kind)
    if kind == "marker":
        (output / assignment.ATTEMPT_MARKER_NAME).write_text("existing", "ascii")
    else:
        (output / assignment.PRIVATE_DIRECTORY_NAME).mkdir()
    with pytest.raises(assignment.OneShotRefusal):
        assignment.run_synthetic_one_shot(
            project,
            output,
            secret_factory=lambda size: FIXED_SECRET,
        )
    assert not (output / assignment.TERMINAL_INCIDENT_NAME).exists()


def test_exception_writes_aggregate_incident_and_second_call_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, output = _fixture(tmp_path, suffix="exception")

    def fail_assignment(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("PRIVATE_TITLE_MUST_NOT_LEAK")

    monkeypatch.setattr(assignment, "_build_assignment_materials", fail_assignment)
    incident = assignment.run_synthetic_one_shot(
        project,
        output,
        secret_factory=lambda size: FIXED_SECRET,
    )
    assert incident["failure_category"] == (
        "implementation_or_infrastructure_exception"
    )
    serialized = _canonical(incident).decode("ascii")
    assert "PRIVATE_TITLE_MUST_NOT_LEAK" not in serialized
    assert not _contains_list(incident)
    _verify_self_hash(incident, "terminal_incident_sha256")
    with pytest.raises(assignment.OneShotRefusal):
        assignment.run_synthetic_one_shot(
            project,
            output,
            secret_factory=lambda size: FIXED_SECRET,
        )


def test_formal_wrong_blob_and_git_provenance_refuse_before_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _unused = _fixture(tmp_path, suffix="formal-preflight")
    base = project.parent
    wrong_project = base / "wrong_project"
    wrong_output = wrong_project / assignment.FORMAL_OUTPUT_RELATIVE_PATH
    (wrong_project / assignment.DESIGN_RELATIVE_PATH.parent).mkdir(parents=True)
    (wrong_project / assignment.DESIGN_RELATIVE_PATH).write_bytes(b"{}\n")
    wrong_output.mkdir(parents=True, mode=0o700)
    wrong_output.chmod(0o700)
    source_calls: list[str] = []

    def forbidden_source(*args: Any, **kwargs: Any) -> bytes:
        source_calls.append("opened")
        raise AssertionError("formal preflight opened a source row")

    monkeypatch.setattr(qualifier, "_read_bound_source", forbidden_source)
    wrong_blob = assignment._run_one_shot_controller(
        wrong_project,
        wrong_output,
        formal_identity_enforced=True,
    )
    assert wrong_blob["failure_category"] == "formal_provenance_invalid"
    assert source_calls == []

    git_output = project / assignment.FORMAL_OUTPUT_RELATIVE_PATH
    git_output.mkdir(parents=True, mode=0o700)
    git_output.chmod(0o700)

    def reject_git(root: Path) -> dict[str, str]:
        raise assignment.FormalProvenanceError("simulated non-ancestor")

    monkeypatch.setattr(assignment, "_validate_formal_provenance", reject_git)
    wrong_git = assignment._run_one_shot_controller(
        project,
        git_output,
        formal_identity_enforced=True,
    )
    assert wrong_git["failure_category"] == "formal_provenance_invalid"
    assert source_calls == []
    assert wrong_git["opened_content_counts"] == {
        "train_annotated_open_count": 0,
        "dev_open_count": 0,
        "relation_metadata_open_count": 0,
        "official_test_open_count": 0,
        "train_distant_open_count": 0,
    }


def test_formal_rejects_secret_injection_and_nonfrozen_output_root(
    tmp_path: Path,
) -> None:
    project, output = _fixture(tmp_path, suffix="formal-boundary")
    with pytest.raises(assignment.FormalProvenanceError):
        assignment._run_one_shot_controller(
            project,
            output,
            formal_identity_enforced=True,
            secret_factory=lambda size: FIXED_SECRET,
        )

    output.chmod(0o700)
    with pytest.raises(assignment.OneShotRefusal, match="frozen path"):
        assignment._run_one_shot_controller(
            project,
            output,
            formal_identity_enforced=True,
        )
