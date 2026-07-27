from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    bioasq_p0_public_source_qualification_v1 as p0,
)
from assumption_agent.benchmarks import bioasq_p1_formal_source_v2 as formal
from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core


FIXED_SECRET = b"bioasq-formal-v2-fixed-secret!!!"
assert len(FIXED_SECRET) == formal.HMAC_SECRET_BYTES


@pytest.fixture
def posix_tmp() -> Iterator[Path]:
    root = Path(
        tempfile.mkdtemp(prefix="bioasq-formal-v2-", dir="/tmp")
    )
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@dataclass(frozen=True)
class _Fixture:
    source: Path
    questions: list[dict[str, Any]]
    p0_receipt: Path
    private_manifest: Path
    contract: formal.FormalSourceContract
    outputs: formal.FormalOutputPaths


def _canonical_file(
    path: Path,
    value: object,
    *,
    mode: int = 0o600,
) -> bytes:
    raw = formal.canonical_bytes(value, newline=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return raw


def _outputs(root: Path, name: str = "outputs") -> formal.FormalOutputPaths:
    output = root / name
    output.mkdir()
    return formal.FormalOutputPaths(
        private_selection_secret=output / "selection.secret.private.bin",
        public_corpus=output / "corpus.public.json",
        public_a_form=output / "A_form.public.json",
        public_f_search=output / "F_search.public.json",
        public_a_hold=output / "A_hold.public.json",
        public_m_search=output / "M_search.public.sealed.json",
        private_a_form_qrels=output / "A_form.qrels.private.json",
        private_a_hold_qrels=output / "A_hold.qrels.private.json",
        private_m_search_qrels=(
            output / "M_search.qrels.private.sealed.json"
        ),
        safe_selection_receipt=output / "selection.safe.json",
    )


def _synthetic_questions(count: int = 1_500) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for index in range(count):
        family = formal.FAMILIES[index % len(formal.FAMILIES)]
        document = f"https://synthetic.invalid/document/{index:04d}"
        body = f"Synthetic biomedical question {index:04d}?"
        snippets = [
            {
                "document": document,
                "text": f"Case-Preserved evidence {index:04d} alpha.",
            },
            {
                "document": document,
                "text": f"Case-preserved evidence {index:04d} beta.",
            },
        ]
        questions.append(
            {
                "body": body,
                "documents": [document],
                "id": f"synthetic-question-{index:04d}",
                "snippets": snippets,
                "type": family,
            }
        )

    # Three independent component-edge fixtures.  They deliberately cross
    # source-native families and leave well over the 56-component demand.
    questions[0]["body"] = "  Shared   Query Alpha? "
    questions[1]["body"] = "shared query alpha?"

    shared_document = "https://synthetic.invalid/document/shared-case"
    questions[2]["documents"] = [shared_document]
    questions[3]["documents"] = [shared_document]
    for index in (2, 3):
        questions[index]["snippets"] = [
            {
                "document": shared_document,
                "text": f"Document-edge evidence {index} alpha.",
            },
            {
                "document": shared_document,
                "text": f"Document-edge evidence {index} beta.",
            },
        ]

    shared_snippet_document = (
        "https://synthetic.invalid/document/shared-snippet"
    )
    shared_snippet = {
        "document": shared_snippet_document,
        "text": "Exact Case-Preserved Shared Snippet.",
    }
    for index in (4, 5):
        questions[index]["documents"] = [shared_snippet_document]
        questions[index]["snippets"] = [
            dict(shared_snippet),
            {
                "document": shared_snippet_document,
                "text": f"Distinct companion evidence {index}.",
            },
        ]
    return questions


def _fixture(tmp_path: Path) -> _Fixture:
    questions = _synthetic_questions()
    raw = json.dumps(
        {"questions": questions},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    source = tmp_path / "synthetic-bioasq-source.json"
    source.write_bytes(raw)
    source.chmod(0o600)
    source_contract = p0.SourceFileContract(
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    qualification_contract = p0.QualificationContract(
        {family: 56 for family in formal.FAMILIES},
        expected_question_count=len(questions),
    )
    source_binding = {
        "file_sha256": source_contract.sha256,
        "size_bytes": source_contract.size_bytes,
        "synthetic_source_free_canary_input": False,
    }
    result = p0._qualify_decoded_source(
        raw,
        source_binding=source_binding,
        contract=qualification_contract,
        source_open_count=1,
        real_source_access_count=1,
    )
    private_manifest = tmp_path / "eligibility.private.json"
    private_raw = _canonical_file(
        private_manifest,
        result.private_manifest,
    )
    p0_receipt = tmp_path / "qualification.safe.json"
    receipt_raw = _canonical_file(
        p0_receipt,
        result.safe_receipt,
    )
    contract = formal.FormalSourceContract(
        source_contract=source_contract,
        qualification_contract=qualification_contract,
        p0_receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        p0_receipt_self_sha256=result.safe_receipt["self_sha256"],
        private_manifest_file_sha256=hashlib.sha256(
            private_raw
        ).hexdigest(),
        private_manifest_self_sha256=(
            result.private_manifest["self_sha256"]
        ),
        p0_implementation_sha256=hashlib.sha256(
            Path(p0.__file__).read_bytes()
        ).hexdigest(),
        typed_core_sha256=hashlib.sha256(
            Path(core.__file__).read_bytes()
        ).hexdigest(),
        block_family_quotas=formal.DEFAULT_BLOCK_FAMILY_QUOTAS,
    )
    return _Fixture(
        source=source,
        questions=questions,
        p0_receipt=p0_receipt,
        private_manifest=private_manifest,
        contract=contract,
        outputs=_outputs(tmp_path),
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _compile(
    fixture: _Fixture,
    *,
    outputs: formal.FormalOutputPaths | None = None,
    secret: bytes = FIXED_SECRET,
) -> dict[str, Any]:
    calls: list[int] = []

    def factory(size: int) -> bytes:
        calls.append(size)
        return secret

    receipt = formal.compile_formal_source(
        p0_receipt_path=fixture.p0_receipt,
        private_eligibility_manifest_path=fixture.private_manifest,
        source_path=fixture.source,
        outputs=fixture.outputs if outputs is None else outputs,
        contract=fixture.contract,
        _secret_factory=factory,
    )
    assert calls == [formal.HMAC_SECRET_BYTES]
    return receipt


def _rebind_source(
    fixture: _Fixture,
    raw: bytes,
) -> _Fixture:
    fixture.source.write_bytes(raw)
    fixture.source.chmod(0o600)
    source_contract = p0.SourceFileContract(
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    source_binding = {
        "file_sha256": source_contract.sha256,
        "size_bytes": source_contract.size_bytes,
        "synthetic_source_free_canary_input": False,
    }

    private = _load(fixture.private_manifest)
    private["source_binding"] = source_binding
    private_body = dict(private)
    private_body.pop("self_sha256")
    private["self_sha256"] = p0.stable_hash(private_body)
    private_raw = _canonical_file(fixture.private_manifest, private)

    receipt = _load(fixture.p0_receipt)
    receipt["source_binding"] = source_binding
    receipt["private_manifest_binding"]["file_sha256"] = hashlib.sha256(
        private_raw
    ).hexdigest()
    receipt["private_manifest_binding"]["self_sha256"] = private[
        "self_sha256"
    ]
    receipt_body = dict(receipt)
    receipt_body.pop("self_sha256")
    receipt["self_sha256"] = p0.stable_hash(receipt_body)
    receipt_raw = _canonical_file(fixture.p0_receipt, receipt)

    contract = replace(
        fixture.contract,
        source_contract=source_contract,
        p0_receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        p0_receipt_self_sha256=receipt["self_sha256"],
        private_manifest_file_sha256=hashlib.sha256(
            private_raw
        ).hexdigest(),
        private_manifest_self_sha256=private["self_sha256"],
    )
    return replace(fixture, contract=contract)


def _assert_self_hash(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    claimed = body.pop("self_sha256")
    assert formal.stable_hash(body) == claimed


def test_single_open_commitment_rebuild_joint_quota_and_fixed_corpus(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(posix_tmp)
    source_opens: list[Path] = []
    original_source_open = formal._open_binary

    def counted_source_open(path: Path) -> Any:
        source_opens.append(path)
        return original_source_open(path)

    monkeypatch.setattr(formal, "_open_binary", counted_source_open)
    secret_exclusive_opens: list[int] = []
    original_os_open = formal.os.open

    def counted_os_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if Path(path).absolute() == (
            fixture.outputs.private_selection_secret.absolute()
        ):
            secret_exclusive_opens.append(flags)
            assert flags & os.O_EXCL
            assert flags & os.O_CREAT
        return original_os_open(path, flags, mode)

    monkeypatch.setattr(formal.os, "open", counted_os_open)
    captured: dict[str, tuple[Any, ...]] = {}
    original_select = formal._select_rows

    def capture_selection(*args: Any, **kwargs: Any) -> Any:
        selected = original_select(*args, **kwargs)
        captured.update(selected)
        return selected

    monkeypatch.setattr(formal, "_select_rows", capture_selection)
    receipt = _compile(fixture)

    assert source_opens == [fixture.source.absolute()]
    assert len(secret_exclusive_opens) == 1
    assert receipt["source_access"] == {
        "file_sha256": fixture.contract.source_contract.sha256,
        "formal_source_access_count": 1,
        "size_bytes": fixture.contract.source_contract.size_bytes,
        "source_hash_count": 1,
        "source_json_decode_count": 1,
        "source_open_count": 1,
    }
    assert receipt["selection"][
        "selection_secret_generation_count"
    ] == 1
    assert receipt["selection"][
        "selection_secret_file_create_count"
    ] == 1
    assert fixture.outputs.private_selection_secret.read_bytes() == (
        FIXED_SECRET
    )
    assert (
        stat.S_IMODE(
            fixture.outputs.private_selection_secret.stat().st_mode
        )
        == 0o600
    )

    selected_rows = [
        row for block in formal.BLOCKS for row in captured[block]
    ]
    assert len(selected_rows) == 224
    assert len(
        {row.component_commitment for row in selected_rows}
    ) == 224
    assert len(
        {row.source.opaque_item_commitment for row in selected_rows}
    ) == 224
    assert len(
        {row.source.query_commitment for row in selected_rows}
    ) == 224
    for block in formal.BLOCKS:
        assert Counter(row.family for row in captured[block]) == Counter(
            formal.DEFAULT_BLOCK_FAMILY_QUOTAS[block]
        )
    assert receipt["disjointness_aggregate"] == {
        "cross_block_component_overlap_count": 0,
        "cross_block_item_overlap_count": 0,
        "cross_block_normalized_query_overlap_count": 0,
        "maximum_selected_items_per_component": 1,
        "selected_component_count": 224,
        "selected_item_count": 224,
        "selected_normalized_query_count": 224,
    }

    # The exact P0 namespaces are rebuilt locally.  Snippet text is
    # intentionally case-preserved rather than casefolded.
    assert formal._p0_commit("normalized_query", "mixed case") == (
        p0._commit("normalized_query", "mixed case")
    )
    assert formal._p0_commit(
        "normalized_gold_snippet",
        "Document\0Case-Preserved",
    ) == p0._commit(
        "normalized_gold_snippet",
        "Document\0Case-Preserved",
    )
    assert formal._p0_commit(
        "normalized_gold_snippet",
        "Document\0Case-Preserved",
    ) != formal._p0_commit(
        "normalized_gold_snippet",
        "Document\0case-preserved",
    )

    corpus = _load(fixture.outputs.public_corpus)
    _assert_self_hash(corpus)
    passages = corpus["passages"]
    assert len(passages) == formal.CORPUS_SIZE
    assert [row["ordinal"] for row in passages] == list(
        range(formal.CORPUS_SIZE)
    )
    assert all(set(row) == formal.PUBLIC_PASSAGE_KEYS for row in passages)
    typed_passages = [
        core.passage_from_public_fields(row) for row in passages
    ]
    assert len(typed_passages) == core.CORPUS_SIZE
    corpus_aggregate = receipt["corpus_aggregate"]
    assert (
        corpus_aggregate["selected_unique_qrel_count"]
        + corpus_aggregate["filler_unique_snippet_count"]
        == formal.CORPUS_SIZE
    )
    assert set(
        corpus_aggregate["arm_corpus_file_sha256"].values()
    ) == {
        receipt["artifact_binding"]["public_corpus"]["file_sha256"]
    }

    work_ids: set[str] = set()
    for block in formal.BLOCKS:
        public = _load(fixture.outputs.public_blocks()[block])
        _assert_self_hash(public)
        assert public["block_id"] == block
        assert len(public["items"]) == sum(
            formal.DEFAULT_BLOCK_FAMILY_QUOTAS[block].values()
        )
        assert all(
            set(item) == formal.PUBLIC_ITEM_KEYS
            for item in public["items"]
        )
        for item in public["items"]:
            assert item["work_id"] not in work_ids
            assert core.validate_query_text(item["query_text"]) == (
                item["query_text"]
            )
            work_ids.add(item["work_id"])
        raw_public = fixture.outputs.public_blocks()[block].read_bytes()
        for forbidden in (
            b'"family":',
            b'"qrel":',
            b'"gold_ordinals":',
            b'"source_id":',
            b'"question_id":',
            b'"document":',
            b'"snippet":',
        ):
            assert forbidden not in raw_public

    for block in formal.QREL_BLOCKS:
        public = _load(fixture.outputs.public_blocks()[block])
        qrels = _load(fixture.outputs.private_qrels()[block])
        _assert_self_hash(qrels)
        assert [item["work_id"] for item in public["items"]] == [
            row["work_id"] for row in qrels["qrels"]
        ]
        assert all(
            set(row) == formal.PRIVATE_QREL_ROW_KEYS
            for row in qrels["qrels"]
        )
        assert all(
            row["family"] in formal.FAMILIES
            and row["gold_ordinals"]
            == sorted(set(row["gold_ordinals"]))
            and len(row["gold_ordinals"]) >= 1
            and all(
                0 <= ordinal < formal.CORPUS_SIZE
                for ordinal in row["gold_ordinals"]
            )
            for row in qrels["qrels"]
        )
    assert len(work_ids) == 224
    assert (
        stat.S_IMODE(
            fixture.outputs.public_m_search.stat().st_mode
        )
        == 0o400
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o400
        for path in fixture.outputs.private_qrels().values()
    )
    assert receipt["seal_contract"][
        "M_search_open_authorization"
    ] == "controller_promotion_authorization_required"
    assert receipt["seal_contract"]["M_search_presealed"] is True


def test_fixed_secret_is_byte_deterministic(posix_tmp: Path) -> None:
    fixture = _fixture(posix_tmp)
    first = _compile(fixture)
    second_outputs = _outputs(posix_tmp, "outputs-second")
    second = _compile(fixture, outputs=second_outputs)
    assert first == second
    for left, right in zip(
        fixture.outputs.all_paths(),
        second_outputs.all_paths(),
    ):
        assert left.read_bytes() == right.read_bytes()


def test_case_preserved_snippet_commitment_drift_is_terminal_and_safe(
    posix_tmp: Path,
) -> None:
    fixture = _fixture(posix_tmp)
    questions = json.loads(json.dumps(fixture.questions))
    original = questions[17]["snippets"][0]["text"]
    questions[17]["snippets"][0]["text"] = original.swapcase()
    raw = json.dumps(
        {"questions": questions},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    fixture = _rebind_source(fixture, raw)

    with pytest.raises(
        formal.BioasqP1FormalSourceError,
        match="complete P0 commitment set",
    ) as captured:
        _compile(fixture)
    assert captured.value.error_code == "p0_commitment_reconstruction"
    failure = _load(fixture.outputs.safe_selection_receipt)
    _assert_self_hash(failure)
    assert failure["status"] == (
        "terminal_formal_source_failure_no_retry"
    )
    assert failure["failure_stage"] == "p0_commitment_reconstruction"
    assert failure["access_boundary"][
        "formal_source_access_count"
    ] == 1
    assert failure["access_boundary"]["source_open_count"] == 1
    assert failure["access_boundary"]["source_hash_count"] == 1
    assert failure["access_boundary"]["source_json_decode_count"] == 1
    assert failure[
        "individual_query_document_snippet_qrel_or_source_id_published"
    ] is False
    assert failure[
        "retry_replay_resample_secret_rotation_source_or_parser_change"
    ] is False


def test_duplicate_key_strict_decode_writes_no_retry_failure_once(
    posix_tmp: Path,
) -> None:
    fixture = _fixture(posix_tmp)
    fixture = _rebind_source(
        fixture,
        b'{"questions":[],"questions":[]}',
    )
    secret_calls: list[int] = []

    def factory(size: int) -> bytes:
        secret_calls.append(size)
        return FIXED_SECRET

    with pytest.raises(
        formal.BioasqP1FormalSourceError,
        match="duplicate object key",
    ) as captured:
        formal.compile_formal_source(
            p0_receipt_path=fixture.p0_receipt,
            private_eligibility_manifest_path=fixture.private_manifest,
            source_path=fixture.source,
            outputs=fixture.outputs,
            contract=fixture.contract,
            _secret_factory=factory,
        )
    assert captured.value.error_code == "strict_json_duplicate_key"
    assert secret_calls == [formal.HMAC_SECRET_BYTES]
    assert fixture.outputs.private_selection_secret.read_bytes() == (
        FIXED_SECRET
    )
    failure_raw = fixture.outputs.safe_selection_receipt.read_bytes()
    failure = json.loads(failure_raw)
    _assert_self_hash(failure)
    assert failure["schema"] == formal.FAILURE_RECEIPT_SCHEMA
    assert failure["status"] == (
        "terminal_formal_source_failure_no_retry"
    )
    assert failure["failure_stage"] == (
        "formal_source_strict_json_decode"
    )
    assert failure["access_boundary"] == {
        "action_count": 0,
        "formal_source_access_count": 1,
        "model_call_count": 0,
        "online_or_API_evaluation_count": 0,
        "score_count": 0,
        "selection_secret_file_create_count": 1,
        "selection_secret_generation_count": 1,
        "source_hash_count": 1,
        "source_json_decode_count": 1,
        "source_open_count": 1,
    }
    assert not fixture.outputs.public_corpus.exists()
    assert not any(
        path.exists()
        for path in fixture.outputs.public_blocks().values()
    )
    assert not any(
        path.exists()
        for path in fixture.outputs.private_qrels().values()
    )

    # The secret and safe terminal burn the attempt.  A second invocation
    # cannot rotate the secret, replace the receipt, or reach source access.
    with pytest.raises(
        formal.BioasqP1FormalSourceError,
        match="not fresh",
    ):
        formal.compile_formal_source(
            p0_receipt_path=fixture.p0_receipt,
            private_eligibility_manifest_path=fixture.private_manifest,
            source_path=fixture.source,
            outputs=fixture.outputs,
            contract=fixture.contract,
            _secret_factory=factory,
        )
    assert secret_calls == [formal.HMAC_SECRET_BYTES]
    assert fixture.outputs.safe_selection_receipt.read_bytes() == failure_raw


def test_default_contract_binds_recorded_p0_safe_and_private_hashes() -> None:
    assert formal.P0_SAFE_RECEIPT_FILE_SHA256 == (
        "344682626cbe138d73bdabf512aedb57fe8d44e041850fa353c340a07fdc73c1"
    )
    assert formal.P0_SAFE_RECEIPT_SELF_SHA256 == (
        "6ea803504d3ec7c65063b696fbad80cb68d80657dd8990ef817e7d1f4b75364f"
    )
    assert formal.P0_PRIVATE_MANIFEST_FILE_SHA256 == (
        "67a8ee8364fd344d0f49eb85cf775597bece4c8937e1d334c248f174be09b71e"
    )
    assert formal.P0_PRIVATE_MANIFEST_SELF_SHA256 == (
        "3d714f8cbb1c9ffc8bd93a00b0cc27979d8eb8cbc91d6b1ad71e3cd596822183"
    )
    assert formal.DEFAULT_CONTRACT.block_family_quotas == {
        "A_form": {family: 24 for family in formal.FAMILIES},
        "F_search": {family: 8 for family in formal.FAMILIES},
        "A_hold": {family: 12 for family in formal.FAMILIES},
        "M_search": {family: 12 for family in formal.FAMILIES},
    }
    assert formal.DEFAULT_CONTRACT.corpus_size == 2_900
