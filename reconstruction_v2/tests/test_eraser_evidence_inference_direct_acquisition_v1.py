from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_direct_acquisition_v1 as subject,
)


@dataclass(frozen=True)
class SourcePaths:
    project: Path
    archive: Path
    sidecar: Path
    qualification: Path
    design: Path
    freeze: Path
    duplicate_ids: tuple[str, str]


@pytest.fixture
def secure_tmp_path() -> Any:
    # The configured pytest tmp root is on DrvFS, which cannot represent the
    # required POSIX 0700/0600 modes.  Acquisition itself must run on ext4.
    with tempfile.TemporaryDirectory(
        prefix="eraser_direct_acquisition_", dir="/tmp"
    ) as value:
        yield Path(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(subject.canonical_bytes(value))


def _self_hashed(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: subject.stable_hash(body)}


def _doc_tokens(index: int) -> tuple[tuple[str, ...], ...]:
    return tuple((f"token{index}_{sentence}", "evidence") for sentence in range(5))


def _doc_bytes(index: int) -> bytes:
    return "".join(" ".join(row) + "\n" for row in _doc_tokens(index)).encode()


def _annotation(
    *,
    annotation_id: str,
    query: str,
    official_class: str,
    docid: str,
    document_index: int,
) -> dict[str, Any]:
    first = _doc_tokens(document_index)[0]
    return {
        "annotation_id": annotation_id,
        "query": query,
        "classification": official_class,
        "query_type": "effect",
        "docids": [docid],
        "evidences": [
            [
                {
                    "text": list(first),
                    "docid": docid,
                    "start_token": 0,
                    "end_token": 2,
                    "start_sentence": 0,
                    "end_sentence": 1,
                }
            ]
        ],
    }


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            bundle.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def _build_source(tmp_path: Path) -> SourcePaths:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    official_by_family = {
        "SIGNIFICANTLY_DECREASED": "significantly decreased",
        "NO_SIGNIFICANT_DIFFERENCE": "no significant difference",
        "SIGNIFICANTLY_INCREASED": "significantly increased",
    }
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    documents: dict[str, tuple[int, bytes]] = {}
    sidecar_rows: list[tuple[str, str]] = []
    next_index = 1

    def add(
        split: str,
        family: str,
        suffix: str,
        *,
        query: str | None = None,
        shared: tuple[str, int] | None = None,
    ) -> tuple[str, str, int]:
        nonlocal next_index
        annotation_id = f"{split}-{family.lower()}-{suffix}"
        if shared is None:
            index = next_index
            next_index += 1
            docid = f"PMC{100000 + index}"
            documents[docid] = (index, _doc_bytes(index))
        else:
            docid, index = shared
        split_rows[split].append(
            _annotation(
                annotation_id=annotation_id,
                query=query or f"Question {split} {family} {suffix}",
                official_class=official_by_family[family],
                docid=docid,
                document_index=index,
            )
        )
        sidecar_rows.append((annotation_id, docid))
        return annotation_id, docid, index

    first_decreased: tuple[str, int] | None = None
    for split, count in (("train", 28), ("val", 20)):
        for family in subject.FAMILY_ORDER:
            for ordinal in range(count):
                _annotation_id, docid, index = add(
                    split, family, f"base-{ordinal}"
                )
                if split == "train" and family == subject.FAMILY_ORDER[0] and ordinal == 0:
                    first_decreased = (docid, index)
    assert first_decreased is not None
    # Same article, different annotation/family: global selection must keep at
    # most one.  A unique decreased backup makes either HMAC outcome feasible.
    add("train", subject.FAMILY_ORDER[1], "article-conflict", shared=first_decreased)
    add("train", subject.FAMILY_ORDER[0], "article-conflict-backup")
    duplicate_train, _docid, _index = add(
        "train",
        subject.FAMILY_ORDER[2],
        "duplicate-query",
        query="  ＤＵＰＬＩＣＡＴＥ   Query ",
    )
    duplicate_val, _docid, _index = add(
        "val",
        subject.FAMILY_ORDER[2],
        "duplicate-query",
        query="duplicate query",
    )

    members: dict[str, bytes] = {
        "evidence_inference/train.jsonl": b"\n".join(
            subject.canonical_bytes(row) for row in split_rows["train"]
        )
        + b"\n",
        "evidence_inference/val.jsonl": b"\n".join(
            subject.canonical_bytes(row) for row in split_rows["val"]
        )
        + b"\n",
        # Invalid/sentinel TEST payloads prove header routing does not open them.
        "evidence_inference/test.jsonl": b"PRIVATE_TEST_ROW_DO_NOT_OPEN\xff",
        "evidence_inference/docs/PMC999999": b"PRIVATE_TEST_DOC_DO_NOT_OPEN\xff",
    }
    for docid, (_index, raw) in documents.items():
        members[f"evidence_inference/docs/{docid}"] = raw
    archive = project / "source.tar.gz"
    archive.write_bytes(_tar_bytes(members))

    sidecar = project / "prompts.csv"
    sidecar_text = "PromptID,PMCID,Outcome,Intervention,Comparator\n" + "".join(
        f"{prompt},{docid},Outcome {prompt},Intervention {prompt},Comparator {prompt}\n"
        for prompt, docid in sidecar_rows
    )
    sidecar_text += "PRIVATE_TEST_PROMPT,PMC999999,DO_NOT_OPEN,DO_NOT_OPEN,DO_NOT_OPEN\n"
    sidecar.write_text(sidecar_text, encoding="utf-8")

    archive_raw = archive.read_bytes()
    sidecar_raw = sidecar.read_bytes()
    design_template = json.loads(
        (
            Path(__file__).parents[1]
            / "manifests/eraser_evidence_inference_r7_e3_design_v1.json"
        ).read_text()
    )
    design_template["source_binding"]["archive_sha256"] = hashlib.sha256(
        archive_raw
    ).hexdigest()
    design_template["source_binding"]["archive_size"] = len(archive_raw)
    design_template["source_binding"]["prompt_sidecar_sha256"] = hashlib.sha256(
        sidecar_raw
    ).hexdigest()
    design_template.pop("design_sha256")
    design = _self_hashed(design_template, "design_sha256")
    design_path = project / "design.json"
    _write_json(design_path, design)

    qualification_body = {
        "schema": subject.QUALIFICATION_SCHEMA,
        "version": subject.QUALIFICATION_SCHEMA,
        "status": "passed_source_qualification_no_selection",
        "source_binding": {
            "whole_archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
            "whole_archive_size": len(archive_raw),
            "prompt_sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
            "prompt_sidecar_size": len(sidecar_raw),
            "custody_manifest_self_sha256": design["source_binding"][
                "custody_self_sha256"
            ],
            "access_manifest_self_sha256": design["source_binding"][
                "source_access_self_sha256"
            ],
            "prompt_access_manifest_self_sha256": design["source_binding"][
                "prompt_sidecar_access_self_sha256"
            ],
        },
        "opened_content_boundary": {
            "authorized_split_member_count": 2,
            "test_member_content_open_count": 0,
        },
        "cross_split_article_disjointness": {
            "article_disjoint": True,
            "train_validation_article_overlap_count": 0,
        },
        "article_disjoint_capacity": {
            "train": {"exact_article_disjoint_capacity_met": True},
            "val": {"exact_article_disjoint_capacity_met": True},
        },
        "independent_structured_prompt_binding": {
            "missing_match_count": 0,
            "duplicate_or_ambiguous_match_count": 0,
            "query_string_reverse_parsing_used": False,
        },
        "claim_boundary": {
            "selection_secret_opened_or_generated": False,
            "cohort_selected": False,
            "online_or_network_evaluation_used": False,
            "test_member_query_document_label_or_content_opened": False,
        },
    }
    qualification = _self_hashed(
        qualification_body, "qualification_sha256"
    )
    qualification_path = project / "qualification.json"
    _write_json(qualification_path, qualification)

    frozen_files: list[dict[str, str]] = []
    for role in subject.REQUIRED_IMPLEMENTATION_ROLE_REGISTRY:
        component = project / f"frozen_{role}.txt"
        component.write_bytes(f"frozen {role}\n".encode())
        frozen_files.append(
            {
                "relative_path": component.name,
                "role": role,
                "sha256": hashlib.sha256(component.read_bytes()).hexdigest(),
            }
        )
    freeze_body = {
        "schema": subject.IMPLEMENTATION_FREEZE_SCHEMA,
        "version": "v1",
        "status": "frozen_before_source_qualification_or_private_assignment",
        "design_sha256": design["design_sha256"],
        "required_role_registry": list(subject.REQUIRED_IMPLEMENTATION_ROLE_REGISTRY),
        "implementation_binding": {"files": frozen_files},
    }
    freeze = _self_hashed(freeze_body, "implementation_freeze_sha256")
    freeze_path = project / "implementation_freeze.json"
    _write_json(freeze_path, freeze)
    return SourcePaths(
        project=project,
        archive=archive,
        sidecar=sidecar,
        qualification=qualification_path,
        design=design_path,
        freeze=freeze_path,
        duplicate_ids=(duplicate_train, duplicate_val),
    )


def _acquire(paths: SourcePaths, root: Path, secret: bytes = b"s" * 32) -> dict[str, Any]:
    return subject.acquire_once(
        archive_path=paths.archive,
        prompt_sidecar_path=paths.sidecar,
        qualification_receipt_path=paths.qualification,
        design_path=paths.design,
        implementation_freeze_path=paths.freeze,
        project_root=paths.project,
        acquisition_root=root,
        selection_secret=secret,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_one_shot_assignment_excludes_duplicate_groups_and_article_conflicts(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    paths = _build_source(tmp_path)
    root = tmp_path / "acquisition"
    receipt = _acquire(paths, root)
    assignment = _load(root / subject.ASSIGNMENT_RELATIVE)

    assert receipt["total_assignment_count"] == 144
    assert subject.OFFICIAL_CLASSIFICATION_TO_FAMILY == {
        "significantly decreased": "SIGNIFICANTLY_DECREASED",
        "no significant difference": "NO_SIGNIFICANT_DIFFERENCE",
        "significantly increased": "SIGNIFICANTLY_INCREASED",
    }
    assert receipt["duplicate_normalized_query_group_count"] == 1
    assert receipt["duplicate_normalized_query_annotation_exclusion_count"] == 2
    assert receipt["source_access_safe_aggregates"]["test_member_content_open_count"] == 0
    assert assignment["block_order"] == list(subject.BLOCK_ORDER)
    assert assignment["family_order"] == list(subject.FAMILY_ORDER)
    assert assignment["block_counts"] == subject.BLOCK_COUNTS
    assert len({row["article_docid"] for row in assignment["assignments"]}) == 144
    assert len({row["annotation_id"] for row in assignment["assignments"]}) == 144
    assert not set(paths.duplicate_ids).intersection(
        row["annotation_id"] for row in assignment["assignments"]
    )
    assert assignment["implementation_freeze_sha256"] == _load(paths.freeze)[
        "implementation_freeze_sha256"
    ]
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for path in root.rglob("*.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    for block in ("A_form", "F_search"):
        view = _load(root / subject.VIEW_DIRECTORY / f"{block}.private.json")
        assert len(view["items"]) == subject.BLOCK_COUNTS[block]
        assert view["family_gold_annotation_docid_or_test_value_included"] is False
        assert set(view["items"][0]["payload"]) == {
            "query",
            "official_ico",
            "sentence_tokens",
        }
        assert "family" not in subject.canonical_bytes(view["items"]).decode()


def test_same_secret_is_deterministic_and_recovery_does_not_reopen_source(
    secure_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = secure_tmp_path
    paths = _build_source(tmp_path)
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _acquire(paths, root_a, b"d" * 32)
    _acquire(paths, root_b, b"d" * 32)
    assignment_a = _load(root_a / subject.ASSIGNMENT_RELATIVE)
    assignment_b = _load(root_b / subject.ASSIGNMENT_RELATIVE)
    assert assignment_a == assignment_b
    assert subject.derive_a_form_fold_key(acquisition_root=root_a) == (
        subject.derive_a_form_fold_key(acquisition_root=root_b)
    )
    assert len(subject.derive_a_form_fold_key(acquisition_root=root_a)) == 32
    assert subject.load_verified_block_view(
        acquisition_root=root_a, block="A_form"
    )["block"] == "A_form"

    def forbidden_source_reopen(**_kwargs: Any) -> Any:
        raise AssertionError("recovery reopened source")

    monkeypatch.setattr(subject, "_load_private_source", forbidden_source_reopen)
    recovered = _acquire(paths, root_a, b"d" * 32)
    assert recovered["private_assignment_sha256"] == assignment_a[
        "private_assignment_sha256"
    ]
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="secret rotation",
    ):
        _acquire(paths, root_a, b"e" * 32)


def test_stage_permissions_A_hold_M_and_label_capability(secure_tmp_path: Path) -> None:
    tmp_path = secure_tmp_path
    paths = _build_source(tmp_path)
    root = tmp_path / "acquisition"
    receipt = _acquire(paths, root)
    assignment = _load(root / subject.ASSIGNMENT_RELATIVE)
    with pytest.raises(subject.EraserEvidenceInferenceDirectAcquisitionError):
        subject.materialize_late_view_once(
            archive_path=paths.archive,
            prompt_sidecar_path=paths.sidecar,
            acquisition_root=root,
            block="A_hold",
            authorization_path=tmp_path / "absent.json",
        )
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="F_search has no label",
    ):
        subject.materialize_label_pack_once(
            archive_path=paths.archive,
            prompt_sidecar_path=paths.sidecar,
            acquisition_root=root,
            block="F_search",
            label_capability_path=tmp_path / "anything.json",
        )
    with pytest.raises(subject.EraserEvidenceInferenceDirectAcquisitionError):
        subject.materialize_label_pack_once(
            archive_path=paths.archive,
            prompt_sidecar_path=paths.sidecar,
            acquisition_root=root,
            block="A_form",
            label_capability_path=tmp_path / "absent-label.json",
        )
    a_form_view = _load(root / subject.VIEW_DIRECTORY / "A_form.private.json")
    f_search_view = _load(root / subject.VIEW_DIRECTORY / "F_search.private.json")
    dummy = "1" * 64
    a_form_capability = tmp_path / "a_form_label_capability.json"
    _write_json(
        a_form_capability,
        subject.build_label_capability(
            block="A_form",
            private_assignment_sha256=assignment["private_assignment_sha256"],
            public_receipt_sha256=receipt["public_receipt_sha256"],
            label_free_view_sha256=a_form_view["label_free_view_sha256"],
            three_arm_execution_seal_sha256=dummy,
            feature_seal_sha256="2" * 64,
        ),
    )
    a_form_labels = subject.materialize_label_pack_once(
        archive_path=paths.archive,
        prompt_sidecar_path=paths.sidecar,
        acquisition_root=root,
        block="A_form",
        label_capability_path=a_form_capability,
    )
    assert len(a_form_labels["items"]) == 48
    assert set(a_form_labels["items"][0]) == {
        "item_commitment_sha256",
        "family",
        "flattened_gold_sentence_ordinals",
        "validated_groups",
    }
    a_form_label_state = subject.load_verified_label_state(
        acquisition_root=root,
        block="A_form",
        label_capability_path=a_form_capability,
    )
    assert a_form_label_state["label_pack_sha256"] == a_form_labels[
        "label_pack_sha256"
    ]
    assert (
        a_form_label_state[
            "upstream_typed_artifact_content_verified_by_acquisition"
        ]
        is False
    )

    f_seal_payload = subject.build_f_policy_seal(
        private_assignment_sha256=assignment["private_assignment_sha256"],
        public_receipt_sha256=receipt["public_receipt_sha256"],
        a_form_label_free_view_sha256=a_form_view["label_free_view_sha256"],
        f_search_label_free_view_sha256=f_search_view["label_free_view_sha256"],
        a_form_three_arm_execution_seal_sha256="1" * 64,
        f_search_three_arm_execution_seal_sha256="4" * 64,
        a_form_feature_seal_sha256="2" * 64,
        f_search_feature_seal_sha256="6" * 64,
        a_form_label_pack_sha256=a_form_labels["label_pack_sha256"],
        a_form_label_capability_sha256=a_form_label_state[
            "label_capability_sha256"
        ],
        a_form_label_capability_file_sha256=a_form_label_state[
            "label_capability_file_sha256"
        ],
        a_form_label_stage_marker_sha256=a_form_label_state[
            "label_stage_marker_sha256"
        ],
        e3_fit_receipt_sha256="7" * 64,
        f_search_policy_receipt_sha256="8" * 64,
    )
    extra_key_seal = dict(f_seal_payload)
    extra_key_seal.pop("f_policy_seal_sha256")
    extra_key_seal["unexpected_gate"] = True
    extra_key_seal = _self_hashed(extra_key_seal, "f_policy_seal_sha256")
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="exact key schema",
    ):
        subject.verify_f_policy_seal(
            extra_key_seal,
            private_assignment_sha256=assignment["private_assignment_sha256"],
            public_receipt_sha256=receipt["public_receipt_sha256"],
            a_form_label_free_view_sha256=a_form_view["label_free_view_sha256"],
            f_search_label_free_view_sha256=f_search_view["label_free_view_sha256"],
        )
    f_seal = tmp_path / "f_seal.json"
    _write_json(f_seal, f_seal_payload)
    a_hold = subject.materialize_late_view_once(
        archive_path=paths.archive,
        prompt_sidecar_path=paths.sidecar,
        acquisition_root=root,
        block="A_hold",
        authorization_path=f_seal,
    )
    assert a_hold["item_count"] == 30
    assert subject.load_verified_block_view(
        acquisition_root=root, block="A_hold", authorization_path=f_seal
    ) == a_hold

    a_hold_capability = tmp_path / "a_hold_label_capability.json"
    _write_json(
        a_hold_capability,
        subject.build_label_capability(
            block="A_hold",
            private_assignment_sha256=assignment["private_assignment_sha256"],
            public_receipt_sha256=receipt["public_receipt_sha256"],
            label_free_view_sha256=a_hold["label_free_view_sha256"],
            three_arm_execution_seal_sha256="9" * 64,
            feature_seal_sha256="a" * 64,
        ),
    )
    a_hold_labels = subject.materialize_label_pack_once(
        archive_path=paths.archive,
        prompt_sidecar_path=paths.sidecar,
        acquisition_root=root,
        block="A_hold",
        label_capability_path=a_hold_capability,
    )
    a_hold_label_state = subject.load_verified_label_state(
        acquisition_root=root,
        block="A_hold",
        label_capability_path=a_hold_capability,
    )

    promotion_payload = subject.build_a_hold_promotion_seal(
        private_assignment_sha256=assignment["private_assignment_sha256"],
        public_receipt_sha256=receipt["public_receipt_sha256"],
        a_hold_label_free_view_sha256=a_hold["label_free_view_sha256"],
        f_policy_seal_sha256=f_seal_payload["f_policy_seal_sha256"],
        a_hold_three_arm_execution_seal_sha256="9" * 64,
        a_hold_feature_seal_sha256="a" * 64,
        a_hold_label_pack_sha256=a_hold_labels["label_pack_sha256"],
        a_hold_label_capability_sha256=a_hold_label_state[
            "label_capability_sha256"
        ],
        a_hold_label_capability_file_sha256=a_hold_label_state[
            "label_capability_file_sha256"
        ],
        a_hold_label_stage_marker_sha256=a_hold_label_state[
            "label_stage_marker_sha256"
        ],
        a_hold_score_receipt_sha256="d" * 64,
        promotion_decision_sha256="e" * 64,
    )
    promotion = tmp_path / "promotion.json"
    _write_json(promotion, promotion_payload)
    m_search = subject.materialize_late_view_once(
        archive_path=paths.archive,
        prompt_sidecar_path=paths.sidecar,
        acquisition_root=root,
        block="M_search",
        authorization_path=promotion,
    )
    assert m_search["item_count"] == 30
    full = subject.verify_full_acquisition_state(
        acquisition_root=root,
        late_authorization_paths={"A_hold": f_seal, "M_search": promotion},
        label_capability_paths={
            "A_form": a_form_capability,
            "A_hold": a_hold_capability,
        },
    )
    assert full["verified_view_blocks"] == list(subject.BLOCK_ORDER)
    assert full["verified_label_blocks"] == ["A_form", "A_hold"]


def test_freeze_precedes_secret_and_incomplete_epoch_cannot_retry(
    secure_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = secure_tmp_path
    paths = _build_source(tmp_path)
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="forbids caller-supplied",
    ):
        subject.acquire_once(
            archive_path=paths.archive,
            prompt_sidecar_path=paths.sidecar,
            qualification_receipt_path=paths.qualification,
            design_path=paths.design,
            implementation_freeze_path=paths.freeze,
            project_root=paths.project,
            acquisition_root=tmp_path / "formal-not-created",
            selection_secret=b"f" * 32,
            enforce_formal_design_identity=True,
        )
    assert not (tmp_path / "formal-not-created").exists()
    bad_freeze = _load(paths.freeze)
    bad_freeze["implementation_binding"]["files"][0]["sha256"] = "0" * 64
    bad_freeze.pop("implementation_freeze_sha256")
    bad_freeze = _self_hashed(bad_freeze, "implementation_freeze_sha256")
    _write_json(paths.freeze, bad_freeze)
    root = tmp_path / "not-created"
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="listed implementation file",
    ):
        _acquire(paths, root)
    assert not root.exists()

    paths = _build_source(tmp_path / "second")
    root = tmp_path / "burned"
    original = subject._load_private_source

    def fail_after_marker(**_kwargs: Any) -> Any:
        raise subject.EraserEvidenceInferenceDirectAcquisitionError("synthetic failure")

    monkeypatch.setattr(subject, "_load_private_source", fail_after_marker)
    with pytest.raises(subject.EraserEvidenceInferenceDirectAcquisitionError):
        _acquire(paths, root)
    assert (root / subject.MARKER_RELATIVE).exists()
    monkeypatch.setattr(subject, "_load_private_source", original)
    with pytest.raises(subject.EraserEvidenceInferenceDirectAcquisitionError):
        _acquire(paths, root)


def test_label_authorization_chain_rejects_swapped_capability_and_minimal_pack(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    paths = _build_source(tmp_path)
    root = tmp_path / "acquisition"
    receipt = _acquire(paths, root)
    assignment = _load(root / subject.ASSIGNMENT_RELATIVE)
    view = _load(root / subject.VIEW_DIRECTORY / "A_form.private.json")
    external_capability = tmp_path / "a_form.capability.json"
    capability = subject.build_label_capability(
        block="A_form",
        private_assignment_sha256=assignment["private_assignment_sha256"],
        public_receipt_sha256=receipt["public_receipt_sha256"],
        label_free_view_sha256=view["label_free_view_sha256"],
        three_arm_execution_seal_sha256="1" * 64,
        feature_seal_sha256="2" * 64,
    )
    _write_json(external_capability, capability)
    subject.materialize_label_pack_once(
        archive_path=paths.archive,
        prompt_sidecar_path=paths.sidecar,
        acquisition_root=root,
        block="A_form",
        label_capability_path=external_capability,
    )
    assert subject.load_verified_label_state(
        acquisition_root=root, block="A_form"
    )["label_capability_sha256"] == capability["label_capability_sha256"]

    stored_capability_path = (
        root
        / subject.AUTHORIZATION_DIRECTORY
        / "label.A_form.private.json"
    )
    original_capability_raw = stored_capability_path.read_bytes()
    swapped_capability = subject.build_label_capability(
        block="A_form",
        private_assignment_sha256=assignment["private_assignment_sha256"],
        public_receipt_sha256=receipt["public_receipt_sha256"],
        label_free_view_sha256=view["label_free_view_sha256"],
        three_arm_execution_seal_sha256="3" * 64,
        feature_seal_sha256="4" * 64,
    )
    _write_json(stored_capability_path, swapped_capability)
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="stage marker",
    ):
        subject.load_verified_label_state(
            acquisition_root=root, block="A_form"
        )
    stored_capability_path.write_bytes(original_capability_raw)

    label_path = root / subject.LABEL_DIRECTORY / "A_form.private.json"
    original_label_raw = label_path.read_bytes()
    minimal_pack = _self_hashed(
        {
            "schema": f"{subject.VERSION}_label_pack",
            "version": subject.VERSION,
            "block": "A_form",
        },
        "label_pack_sha256",
    )
    _write_json(label_path, minimal_pack)
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="label pack exact key schema",
    ):
        subject.load_verified_label_state(
            acquisition_root=root, block="A_form"
        )
    label_path.write_bytes(original_label_raw)

    out_of_bounds_pack = json.loads(original_label_raw)
    out_of_bounds_pack.pop("label_pack_sha256")
    out_of_bounds_pack["items"][0]["flattened_gold_sentence_ordinals"] = [999]
    out_of_bounds_pack["items"][0]["validated_groups"] = [[999]]
    _write_json(
        label_path,
        _self_hashed(out_of_bounds_pack, "label_pack_sha256"),
    )
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="gold semantics",
    ):
        subject.load_verified_label_state(
            acquisition_root=root, block="A_form"
        )
    label_path.write_bytes(original_label_raw)
    assert subject.load_verified_label_state(
        acquisition_root=root,
        block="A_form",
        label_capability_path=external_capability,
    )["label_pack_sha256"] == _load(label_path)["label_pack_sha256"]


def test_public_and_label_free_views_reject_rehashed_extra_leak_fields(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    paths = _build_source(tmp_path)
    root = tmp_path / "acquisition"
    _acquire(paths, root)

    view_path = root / subject.VIEW_DIRECTORY / "A_form.private.json"
    view_raw = view_path.read_bytes()
    view = _load(view_path)
    view.pop("label_free_view_sha256")
    view["family"] = "synthetic-leak"
    _write_json(view_path, _self_hashed(view, "label_free_view_sha256"))
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="label-free view exact key schema",
    ):
        subject.load_verified_block_view(
            acquisition_root=root, block="A_form"
        )
    view_path.write_bytes(view_raw)

    public_path = root / subject.PUBLIC_RECEIPT_RELATIVE
    public = _load(public_path)
    public.pop("public_receipt_sha256")
    public["annotation_id"] = "synthetic-leak"
    _write_json(public_path, _self_hashed(public, "public_receipt_sha256"))
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="public acquisition receipt exact key schema",
    ):
        subject.verify_acquisition_state(acquisition_root=root)


def test_assignment_tamper_fails_closed(secure_tmp_path: Path) -> None:
    tmp_path = secure_tmp_path
    paths = _build_source(tmp_path)
    root = tmp_path / "acquisition"
    _acquire(paths, root)
    assignment_path = root / subject.ASSIGNMENT_RELATIVE
    assignment = _load(assignment_path)
    assignment["assignments"][0]["family"] = subject.FAMILY_ORDER[1]
    assignment_path.write_bytes(subject.canonical_bytes(assignment))
    with pytest.raises(
        subject.EraserEvidenceInferenceDirectAcquisitionError,
        match="self-hash drifted",
    ):
        subject.verify_acquisition_state(acquisition_root=root)
