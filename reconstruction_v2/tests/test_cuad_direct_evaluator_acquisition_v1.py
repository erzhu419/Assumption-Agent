from __future__ import annotations

from collections import Counter
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest

from assumption_agent.benchmarks import cuad_direct_evaluator_acquisition_v1 as c


def _context(index: int) -> str:
    return (
        f"section 1. Alpha clause {index}.\n\n"
        f"section 2. Beta clause {index}.\n\n"
        f"section 3. Gamma clause {index}.\n\n"
        f"section 4. Delta clause {index}.\n\n"
        f"section 5. Epsilon clause {index}."
    )


def _entry(index: int, *, title: str | None = None, context: str | None = None) -> dict[str, object]:
    context = _context(index) if context is None else context
    answer = f"Alpha clause {index}"
    start = context.index(answer)
    return {
        "title": title if title is not None else f"Private contract {index}",
        "paragraphs": [
            {
                "context": context,
                "qas": [
                    {
                        "id": f"private-{index}",
                        "question": f"Which alpha clause belongs to contract {index}?",
                        "is_impossible": False,
                        "answers": [{"text": answer, "answer_start": start}],
                    }
                ],
            }
        ],
    }


def _payload(count: int) -> dict[str, object]:
    return {"version": "synthetic", "data": [_entry(index) for index in range(count)]}


def _write_zip(
    path: Path,
    payload: object,
    *,
    member: str = "nested/train_separate_questions.json",
    extras: list[tuple[str | zipfile.ZipInfo, bytes]] | None = None,
) -> tuple[Path, bytes]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, raw)
        archive.writestr("CUADv1.json", b"forbidden decoy")
        archive.writestr("test.json", b"forbidden test decoy")
        for name, value in extras or []:
            archive.writestr(name, value)
    return path, raw


def _paths(tmp_path: Path) -> c.OutputPaths:
    private = {
        block: (
            tmp_path / "private" / f"{block}.views.json",
            tmp_path / "private" / f"{block}.labels.json",
        )
        for block in c.BLOCK_ORDER
    }
    return c.OutputPaths(
        marker=tmp_path / "custody" / "attempt.marker",
        failure=tmp_path / "custody" / "failure.json",
        public_receipt=tmp_path / "public" / "receipt.json",
        private=private,
    )


def _run_synthetic(tmp_path: Path, payload: object, *, secret: bytes = b"s" * 32):
    archive, _raw = _write_zip(tmp_path / "source.zip", payload)
    binding = c.hash_archive(archive)
    member = c.inspect_zip_central_directory(archive)
    paths = _paths(tmp_path)
    receipt = c.execute_acquisition_once(
        archive_path=archive,
        archive_binding=binding,
        bound_member=member,
        secret=secret,
        protocol_bindings={"synthetic_fixture": True},
        paths=paths,
    )
    return receipt, paths, archive, binding, member


def _record(
    ordinal: int, *, title: str, context: str, qas: tuple[object, ...] = ()
) -> c.ParagraphRecord:
    normalized_title = c.exposure_normalize(title)
    normalized_context = c.exposure_normalize(context)
    return c.ParagraphRecord(
        ordinal=ordinal,
        title=title,
        normalized_title=normalized_title,
        normalized_title_sha256=hashlib.sha256(
            normalized_title.encode("utf-8")
        ).hexdigest(),
        context=context,
        normalized_context_sha256=hashlib.sha256(
            normalized_context.encode("utf-8")
        ).hexdigest(),
        raw_context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
        qas=qas,
    )


def test_segmenter_preserves_offsets_hard_starts_and_fallback() -> None:
    context = (
        "  section 1. Alpha.\n\n"
        " (a) Beta;\n\n"
        "clause 2. Gamma:\n\n"
        "paragraph 3. Delta.\n\n"
        "(e) Epsilon.  "
    )
    nodes = c.segment_context(context)
    assert len(nodes) == 5
    assert [node.span_i for node in nodes] == list(range(5))
    assert all(node.identity_text == context[node.start : node.end] for node in nodes)
    assert all(not node.identity_text[0].isspace() for node in nodes)
    assert all(not node.identity_text[-1].isspace() for node in nodes)

    fallback = "x" * 1099 + " " + "y" * 400
    fallback_nodes = c.segment_context(fallback)
    assert [(node.start, node.end) for node in fallback_nodes] == [(0, 1099), (1100, 1500)]
    assert max(len(node.identity_text) for node in fallback_nodes) <= c.MAX_NODE_CHARS


def test_answer_offsets_and_strict_omitted_alignment() -> None:
    context = "Alpha Beta Beta"
    one_node = (c.SourceNode(0, 0, len(context), context),)
    assert c.map_answer_to_nodes(
        context=context, nodes=one_node, text="Alpha", answer_start=0
    ) == frozenset({0})
    assert c.map_answer_to_nodes(
        context=context,
        nodes=one_node,
        text="Alpha<omitted>Beta",
        answer_start=0,
    ) == frozenset({0})

    split_nodes = (
        c.SourceNode(0, 0, 5, "Alpha"),
        c.SourceNode(1, 6, 10, "Beta"),
        c.SourceNode(2, 11, 15, "Beta"),
    )
    with pytest.raises(c.LocalRowError, match="omitted_alignment_ambiguous"):
        c.map_answer_to_nodes(
            context=context,
            nodes=split_nodes,
            text="Alpha<omitted>Beta",
            answer_start=0,
        )
    with pytest.raises(c.LocalRowError, match="answer_offset_mismatch"):
        c.map_answer_to_nodes(
            context=context, nodes=split_nodes, text="Alpha", answer_start=1
        )
    with pytest.raises(c.LocalRowError, match="answer_schema"):
        c.map_answer_to_nodes(
            context=context, nodes=split_nodes, text="Alpha", answer_start=True
        )


def test_dsu_joins_by_title_or_normalized_context_transitively() -> None:
    records = (
        _record(0, title="Same Title", context="A"),
        _record(1, title=" same   title ", context="B"),
        _record(2, title="Other", context=" B "),
        _record(3, title="Independent", context="C"),
    )
    components = c.build_components(records)
    assert [[row.ordinal for row in component] for component in components] == [
        [0, 1, 2],
        [3],
    ]


def test_component_commitment_and_hmac_qa_choice_follow_custody_arrays() -> None:
    secret = bytes(range(32))
    entry = _entry(3)
    paragraph = entry["paragraphs"][0]
    assert isinstance(paragraph, dict)
    qas = paragraph["qas"]
    assert isinstance(qas, list)
    first = dict(qas[0])
    second = dict(first)
    first["id"] = "qa-z"
    second["id"] = "qa-a"
    second["question"] = "Second valid question"
    paragraph["qas"] = [first, second]

    blocks, stats = c.select_blocks_from_payload({"data": [entry]}, secret=secret)
    assert stats["component_counts"]["eligible"] == 1
    selected = blocks["A_form"][0]
    title_norm = c.exposure_normalize(str(entry["title"]))
    context = str(paragraph["context"])
    title_sha = hashlib.sha256(title_norm.encode()).hexdigest()
    context_sha = hashlib.sha256(c.exposure_normalize(context).encode()).hexdigest()
    component_raw = json.dumps(
        ["cuad_direct_v1", "component", [title_sha], [context_sha]],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    expected_component = hashlib.sha256(component_raw).hexdigest()
    assert selected.component_commitment_sha256 == expected_component

    candidates = []
    for qa_id in ("qa-z", "qa-a"):
        qa_sha = hashlib.sha256(qa_id.encode()).hexdigest()
        message = json.dumps(
            ["cuad_direct_v1", "item", expected_component, qa_sha],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        candidates.append((hmac.new(secret, message, hashlib.sha256).digest(), qa_sha))
    assert selected.exact_qa_id_sha256 == min(candidates)[1]


def test_full_exposure_denylist_excludes_entire_component() -> None:
    exposed_title = _entry(1, title=c.EXPOSED_TITLE_OR_ID_PREFIX + " appendix")
    exposed_context = _entry(
        2,
        context=_context(2)
        + "\n\n"
        + c.EXPOSED_CONTEXT_SIGNATURES[0].replace("'", "\u2019"),
    )
    exposed_id = _entry(3)
    qa = exposed_id["paragraphs"][0]["qas"][0]
    qa["id"] = c.EXPOSED_TITLE_OR_ID_PREFIX + "-qa"
    safe = _entry(4)
    payload = {"data": [exposed_title, exposed_context, exposed_id, safe]}
    blocks, stats = c.select_blocks_from_payload(payload, secret=b"e" * 32)
    assert sum(len(rows) for rows in blocks.values()) == 1
    assert stats["component_counts"] == {
        "constructed": 4,
        "exposure_excluded": 3,
        "eligible": 1,
        "required": 256,
        "capacity_satisfied": False,
    }
    assert stats["exposure_counts"] == {"title_or_id": 2, "context_signature": 1}


def test_zip_central_directory_is_safe_and_does_not_open_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _raw = _write_zip(tmp_path / "safe.zip", _payload(1))
    original_open = zipfile.ZipFile.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("central-directory inspection opened a member")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_open)
    member = c.inspect_zip_central_directory(archive)
    assert member.path == "nested/train_separate_questions.json"
    monkeypatch.setattr(zipfile.ZipFile, "open", original_open)

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as handle:
        handle.writestr("../train_separate_questions.json", b"{}")
    with pytest.raises(c.CUADAcquisitionError, match="unsafe member path"):
        c.inspect_zip_central_directory(unsafe)

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("train_separate_questions.json")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as handle:
        handle.writestr(info, b"target")
    with pytest.raises(c.CUADAcquisitionError, match="symlink or nonregular"):
        c.inspect_zip_central_directory(symlink)

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as handle:
        handle.writestr("a/train_separate_questions.json", b"{}")
        handle.writestr("b/train_separate_questions.json", b"{}")
    with pytest.raises(c.CUADAcquisitionError, match="basename is not unique"):
        c.inspect_zip_central_directory(duplicate)


def test_formal_api_cannot_be_called_as_a_row_probe(tmp_path: Path) -> None:
    with pytest.raises(c.CUADAcquisitionError, match="only through --formal"):
        c.formal_acquire(
            project=tmp_path,
            archive_path=tmp_path / "x.zip",
            selection_secret_path=tmp_path / "secret",
            output_path=tmp_path / "out.json",
        )


def test_marker_precedes_member_read_no_replay_and_capacity_has_no_private_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _raw = _write_zip(tmp_path / "source.zip", _payload(2))
    binding = c.hash_archive(archive)
    member = c.inspect_zip_central_directory(archive)
    paths = _paths(tmp_path)
    original_read = c._read_bound_member
    observed = {"called": 0}

    def checked_read(path, bound, initial):
        observed["called"] += 1
        assert paths.marker.is_file()
        assert stat.S_IMODE(paths.marker.stat().st_mode) == 0o600
        marker = json.loads(paths.marker.read_text(encoding="utf-8"))
        assert marker["TRAIN_member_opened_before_marker"] is False
        return original_read(path, bound, initial)

    monkeypatch.setattr(c, "_read_bound_member", checked_read)
    receipt = c.execute_acquisition_once(
        archive_path=archive,
        archive_binding=binding,
        bound_member=member,
        secret=b"m" * 32,
        protocol_bindings={"synthetic_fixture": True},
        paths=paths,
    )
    assert observed["called"] == 1
    assert receipt["status"] == "terminal_source_capacity_insufficient"
    assert paths.public_receipt.is_file()
    assert not any(path.exists() for pair in paths.private.values() for path in pair)

    with pytest.raises(c.CUADAcquisitionError, match="output already exists"):
        c.execute_acquisition_once(
            archive_path=archive,
            archive_binding=binding,
            bound_member=member,
            secret=b"m" * 32,
            protocol_bindings={"synthetic_fixture": True},
            paths=paths,
        )
    assert observed["called"] == 1


def test_full_256_selection_is_disjoint_sealed_and_publicly_redacted(tmp_path: Path) -> None:
    payload = _payload(257)
    receipt, paths, _archive, _binding, _member = _run_synthetic(
        tmp_path, payload, secret=bytes(range(32))
    )
    assert receipt["status"] == "private_four_block_pack_formed"
    assert receipt["blocks"]["selected_item_count"] == 256
    assert receipt["blocks"]["global_component_disjointness"] is True
    assert len(receipt["blocks"]["private_file_commitments"]) == 4

    all_components: list[str] = []
    for block in c.BLOCK_ORDER:
        view_path, label_path = paths.private[block]
        assert stat.S_IMODE(view_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(label_path.stat().st_mode) == 0o600
        views = json.loads(view_path.read_text(encoding="utf-8"))
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        assert views["schema"] == c.LABEL_FREE_BLOCK_SCHEMA
        assert labels["schema"] == c.LABEL_BLOCK_SCHEMA
        assert views["count"] == labels["count"] == 64
        for ordinal, (view, label) in enumerate(zip(views["rows"], labels["rows"], strict=True)):
            assert view["ordinal"] == label["ordinal"] == ordinal
            assert view["item_commitment_sha256"] == label["item_commitment_sha256"]
            assert "gold_node_indices" not in view
            assert set(label) == {
                "schema",
                "block",
                "ordinal",
                "item_commitment_sha256",
                "gold_node_indices",
            }
            assert all(
                node["identity_text"]
                == "".join([node["identity_text"]])
                for node in view["nodes"]
            )
            all_components.append(view["component_commitment_sha256"])
    assert len(all_components) == len(set(all_components)) == 256

    public_raw = paths.public_receipt.read_text(encoding="utf-8")
    public = json.loads(public_raw)
    declared = public.pop("acquisition_sha256")
    assert declared == hashlib.sha256(
        json.dumps(
            public,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert "Which alpha clause belongs" not in public_raw
    assert "Private contract" not in public_raw
    assert "Alpha clause" not in public_raw
    assert "forbidden decoy" not in public_raw
    assert "forbidden test decoy" not in public_raw


def test_post_marker_root_failure_is_terminal_and_not_replayable(tmp_path: Path) -> None:
    archive, _raw = _write_zip(tmp_path / "source.zip", ["not", "a", "root"])
    binding = c.hash_archive(archive)
    member = c.inspect_zip_central_directory(archive)
    paths = _paths(tmp_path)
    with pytest.raises(c.CUADAcquisitionError, match="SQuAD-v2 envelope"):
        c.execute_acquisition_once(
            archive_path=archive,
            archive_binding=binding,
            bound_member=member,
            secret=b"f" * 32,
            protocol_bindings={"synthetic_fixture": True},
            paths=paths,
        )
    assert paths.marker.is_file()
    assert paths.failure.is_file()
    public = json.loads(paths.public_receipt.read_text(encoding="utf-8"))
    assert public["status"] == "terminal_infrastructure_invalid"
    assert public["aggregate"]["same_source_replay_authorized"] is False
    with pytest.raises(c.CUADAcquisitionError, match="output already exists"):
        c.execute_acquisition_once(
            archive_path=archive,
            archive_binding=binding,
            bound_member=member,
            secret=b"f" * 32,
            protocol_bindings={"synthetic_fixture": True},
            paths=paths,
        )


def test_local_qa_errors_are_counted_without_root_abort() -> None:
    entry = _entry(9)
    paragraph = entry["paragraphs"][0]
    valid = dict(paragraph["qas"][0])
    bad_offset = dict(valid)
    bad_offset["id"] = "bad-offset"
    bad_offset["answers"] = [{"text": "Alpha", "answer_start": 99_999}]
    impossible = dict(valid)
    impossible["id"] = "impossible"
    impossible["is_impossible"] = True
    duplicate_a = dict(valid)
    duplicate_a["id"] = "duplicate"
    duplicate_b = dict(valid)
    duplicate_b["id"] = "duplicate"
    paragraph["qas"] = [valid, bad_offset, impossible, duplicate_a, duplicate_b]
    blocks, stats = c.select_blocks_from_payload({"data": [entry]}, secret=b"q" * 32)
    assert sum(len(rows) for rows in blocks.values()) == 1
    reasons = stats["parser_reason_counts"]
    assert reasons["answer_offset_mismatch"] == 1
    assert reasons["qa_impossible"] == 1
    assert reasons["duplicate_qa_id"] == 2

