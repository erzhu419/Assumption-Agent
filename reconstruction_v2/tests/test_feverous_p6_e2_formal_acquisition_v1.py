from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_p6_e2_formal_acquisition_v1 as subject


IMPLEMENTATION_SHA = "a" * 64
EQUIVALENCE_SHA = "b" * 64
SECRET = b"S" * 32


def _payloads() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    corpus = acquisition.self_hashed(
        {
            "schema": acquisition.CORPUS_VIEW_SCHEMA,
            "version": acquisition.VERSION,
            "unit_count": acquisition.CORPUS_UNIT_COUNT,
            "gold_origin_or_membership_included": False,
            "units": [],
        },
        "corpus_view_sha256",
    )
    views: dict[str, dict[str, object]] = {}
    labels: dict[str, dict[str, object]] = {}
    for block in acquisition.BLOCK_ORDER:
        views[block] = acquisition.self_hashed(
            {
                "schema": acquisition.BLOCK_VIEW_SCHEMA,
                "version": acquisition.VERSION,
                "item_count": acquisition.BLOCK_COUNTS[block],
                "late_label_fields_included": False,
                "items": [
                    {"claim": f"opaque claim {ordinal}"}
                    for ordinal in range(acquisition.BLOCK_COUNTS[block])
                ],
            },
            "block_view_sha256",
        )
        if block != "F_search":
            labels[block] = acquisition.self_hashed(
                {
                    "schema": acquisition.BLOCK_LABEL_SCHEMA,
                    "version": acquisition.VERSION,
                    "block": block,
                    "item_count": acquisition.BLOCK_COUNTS[block],
                    "items": [
                        {
                            "ordinal": ordinal,
                            "gold_unit_indices": [0, 1],
                            "family": acquisition.FAMILIES[
                                ordinal % len(acquisition.FAMILIES)
                            ],
                            "verdict": acquisition.VERDICTS[ordinal % 2],
                        }
                        for ordinal in range(acquisition.BLOCK_COUNTS[block])
                    ],
                },
                "block_labels_sha256",
            )
    return corpus, views, labels


def _patch_success_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_during_materialization: bool = False,
) -> list[str]:
    events: list[str] = []

    class FakeSource:
        def __init__(self, **_kwargs: object) -> None:
            events.append("source_init")
            self.identity_exhausted = False
            self.selected_exhausted = False

        def __enter__(self) -> "FakeSource":
            events.append("source_enter")
            return self

        def __exit__(self, *_exc: object) -> None:
            events.append("source_exit")

        def read_annotations_once(self) -> tuple[dict[str, object], ...]:
            events.append("annotations_once")
            return ({"train": True},)

        def exact_resolver_for_candidate_screen(self) -> object:
            events.append("candidate_resolver_once")
            return object()

        def plan_corpus_identities_parallel_once(
            self, **kwargs: object
        ) -> object:
            assert kwargs["blocks"] is blocks
            assert kwargs["secret"] == SECRET
            assert (
                kwargs[
                    "identity_full_compile_equivalence_qualification_sha256"
                ]
                == EQUIVALENCE_SHA
            )
            events.append("parallel_identity_exact_cover_open")
            self.identity_exhausted = True
            events.append("parallel_identity_exact_cover_exhausted")
            return plan

        def iter_selected_corpus_units_once(self, _plan: object) -> object:
            assert self.identity_exhausted
            events.append("selected_stream_open")

            def rows() -> object:
                events.append("selected_stream_first")
                yield "unit"
                self.selected_exhausted = True
                events.append("selected_stream_exhausted")

            return rows()

        @property
        def annotation_receipt(self) -> dict[str, object]:
            return {"receipt": "annotation"}

        @property
        def database_receipt(self) -> dict[str, object]:
            assert self.identity_exhausted
            return {"receipt": "database_exhausted"}

        @property
        def selected_lookup_receipt(self) -> dict[str, object]:
            assert self.selected_exhausted
            return {"receipt": "selected_exhausted"}

    def adapt(records: object, **_kwargs: object) -> SimpleNamespace:
        assert tuple(records) == ({"train": True},)
        events.append("adapt_once")
        return SimpleNamespace(
            candidates=("candidate",),
            receipt={"receipt": "adapter"},
        )

    blocks = {
        block: (f"{block}_assigned",) for block in acquisition.BLOCK_ORDER
    }

    def select(candidates: object, secret: bytes) -> tuple[object, object]:
        assert tuple(candidates) == ("candidate",)
        assert secret == SECRET
        events.append("select_all_four_once")
        return blocks, {"selection": "aggregate"}

    plan = SimpleNamespace(name="single_plan")

    corpus_stats = {"corpus": "aggregate"}

    def materialize_corpus(
        *,
        plan: object,
        units: object,
        secret: bytes,
        require_formal_source: bool,
    ) -> tuple[object, object, object]:
        assert getattr(plan, "name", None) == "single_plan"
        assert tuple(units) == ("unit",)
        assert secret == SECRET
        assert require_formal_source is True
        events.append("materialize_after_selected_exhaustion")
        if fail_during_materialization:
            raise RuntimeError(SECRET.hex())
        return (), {}, corpus_stats

    payloads = _payloads()

    def materialize_payloads(**kwargs: object) -> object:
        assert kwargs["blocks"] is blocks
        events.append("all_four_payloads_one_call")
        return payloads

    monkeypatch.setattr(subject.secrets, "token_bytes", lambda size: SECRET)
    monkeypatch.setattr(subject.formal_source, "ControlledTrainSource", FakeSource)
    monkeypatch.setattr(subject.source_adapter, "adapt_train_candidate_records", adapt)
    monkeypatch.setattr(subject.source_adapter, "verify_adapter_receipt", lambda _r: None)
    monkeypatch.setattr(subject.acquisition, "select_private_blocks", select)
    monkeypatch.setattr(
        subject.acquisition,
        "materialize_fixed_corpus_from_selection_plan",
        materialize_corpus,
    )
    monkeypatch.setattr(
        subject.acquisition,
        "materialize_private_payloads",
        materialize_payloads,
    )
    monkeypatch.setattr(
        subject.acquisition, "verify_formal_corpus_acquisition", lambda _r: None
    )
    monkeypatch.setattr(
        subject.formal_source, "verify_annotation_receipt", lambda _r: None
    )
    monkeypatch.setattr(
        subject.formal_source,
        "require_formal_database_page_stream_receipt",
        lambda _r: None,
    )
    monkeypatch.setattr(
        subject.formal_source,
        "verify_selected_page_lookup_receipt",
        lambda _r: None,
    )
    return events


def _acquire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events = _patch_success_pipeline(monkeypatch)
    receipt = subject.perform_formal_acquisition_once(
        project=tmp_path,
        implementation_freeze_sha256=IMPLEMENTATION_SHA,
        identity_full_compile_equivalence_qualification_sha256=EQUIVALENCE_SHA,
    )
    assert receipt == subject.verify_acquisition_receipt(tmp_path)
    return events


def _rewrite_receipt(tmp_path: Path, receipt: dict[str, object]) -> None:
    body = dict(receipt)
    body.pop("acquisition_receipt_sha256", None)
    receipt = {**body, "acquisition_receipt_sha256": subject._semantic_hash(body)}
    path = tmp_path / subject.RECEIPT_RELATIVE
    path.write_bytes(subject._canonical_bytes(receipt))
    os.chmod(path, 0o600)


def test_one_acquisition_forms_all_four_claim_only_packs_and_no_F_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _acquire(tmp_path, monkeypatch)
    paths = subject.acquisition_paths(tmp_path)
    assert events.count("select_all_four_once") == 1
    assert events.count("all_four_payloads_one_call") == 1
    assert events.index("parallel_identity_exact_cover_exhausted") < events.index(
        "selected_stream_open"
    )
    assert events.index("selected_stream_exhausted") < events.index(
        "all_four_payloads_one_call"
    )
    assert all(path.is_file() for path in paths.views.values())
    assert set(paths.labels) == {"A_form", "A_hold", "M_search"}
    assert not (tmp_path / subject.F_SEARCH_LABEL_RELATIVE).exists()
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in (
            paths.marker,
            paths.receipt,
            paths.secret,
            paths.corpus,
            *paths.views.values(),
            *paths.labels.values(),
        )
    )
    for block in acquisition.BLOCK_ORDER:
        view = subject.load_block_view(tmp_path, block=block)
        assert "block" not in view
        assert all(set(row) == {"claim"} for row in view["items"])

    public = paths.receipt.read_bytes()
    assert SECRET not in public
    assert SECRET.hex().encode("ascii") not in public
    decoded = json.loads(public)
    assert decoded["selection_secret_persisted_publicly"] is False
    assert decoded["F_search_gold_pack_created"] is False
    assert decoded["all_blocks_one_acquisition"] is True
    assert subject.load_private_secret(tmp_path) == SECRET

    before = {path: path.read_bytes() for path in (paths.receipt, paths.secret)}
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.perform_formal_acquisition_once(
            project=tmp_path,
            implementation_freeze_sha256=IMPLEMENTATION_SHA,
            identity_full_compile_equivalence_qualification_sha256=EQUIVALENCE_SHA,
        )
    assert {path: path.read_bytes() for path in before} == before
    assert not paths.failure.exists()


def test_partial_failure_is_terminal_secret_safe_and_not_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_success_pipeline(monkeypatch, fail_during_materialization=True)
    with pytest.raises(RuntimeError):
        subject.perform_formal_acquisition_once(
            project=tmp_path,
            implementation_freeze_sha256=IMPLEMENTATION_SHA,
            identity_full_compile_equivalence_qualification_sha256=EQUIVALENCE_SHA,
        )
    paths = subject.acquisition_paths(tmp_path)
    assert paths.marker.is_file()
    assert paths.secret.read_bytes() == SECRET
    assert paths.failure.is_file()
    assert not paths.receipt.exists()
    failure = paths.failure.read_bytes()
    assert SECRET not in failure
    assert SECRET.hex().encode("ascii") not in failure
    before = failure
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.perform_formal_acquisition_once(
            project=tmp_path,
            implementation_freeze_sha256=IMPLEMENTATION_SHA,
            identity_full_compile_equivalence_qualification_sha256=EQUIVALENCE_SHA,
        )
    assert paths.failure.read_bytes() == before


def test_private_pack_secret_F_label_and_marker_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _acquire(tmp_path, monkeypatch)
    paths = subject.acquisition_paths(tmp_path)
    original_view = paths.views["A_form"].read_bytes()
    paths.views["A_form"].write_bytes(original_view + b" ")
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.verify_acquisition_receipt(tmp_path)
    paths.views["A_form"].write_bytes(original_view)

    os.chmod(paths.views["A_form"], 0o640)
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.verify_acquisition_receipt(tmp_path)
    os.chmod(paths.views["A_form"], 0o600)

    original_secret = paths.secret.read_bytes()
    paths.secret.write_bytes(b"T" * 32)
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.verify_acquisition_receipt(tmp_path)
    paths.secret.write_bytes(original_secret)

    f_label = tmp_path / subject.F_SEARCH_LABEL_RELATIVE
    f_label.write_bytes(b"{}\n")
    os.chmod(f_label, 0o600)
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.verify_acquisition_receipt(tmp_path)
    f_label.unlink()

    marker = json.loads(paths.marker.read_text("ascii"))
    body = dict(marker)
    body.pop("marker_sha256")
    body["implementation_freeze_sha256"] = "c" * 64
    marker = {**body, "marker_sha256": subject._semantic_hash(body)}
    paths.marker.write_bytes(subject._canonical_bytes(marker))
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.verify_acquisition_receipt(tmp_path)


def test_rehashed_role_path_swap_and_receipt_extension_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _acquire(tmp_path, monkeypatch)
    path = tmp_path / subject.RECEIPT_RELATIVE
    receipt = json.loads(path.read_text("ascii"))
    rows = [dict(row) for row in receipt["private_file_bindings"]]
    left = next(index for index, row in enumerate(rows) if row["role"] == "A_form_view")
    right = next(index for index, row in enumerate(rows) if row["role"] == "F_search_view")
    rows[left]["role"], rows[right]["role"] = rows[right]["role"], rows[left]["role"]
    receipt["private_file_bindings"] = rows
    receipt["private_file_binding_set_sha256"] = subject._semantic_hash(rows)
    _rewrite_receipt(tmp_path, receipt)
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.verify_acquisition_receipt(tmp_path)

    # Restore a fresh valid semantic receipt, then prove that an unregistered
    # extension cannot be made valid merely by recomputing the public hash.
    rows[left]["role"], rows[right]["role"] = rows[right]["role"], rows[left]["role"]
    receipt["private_file_bindings"] = rows
    receipt["private_file_binding_set_sha256"] = subject._semantic_hash(rows)
    receipt["unexpected"] = hashlib.sha256(b"extension").hexdigest()
    _rewrite_receipt(tmp_path, receipt)
    with pytest.raises(subject.FeverousFormalAcquisitionError):
        subject.verify_acquisition_receipt(tmp_path)


def test_staged_verification_never_reads_future_sealed_role_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _acquire(tmp_path, monkeypatch)
    paths = subject.acquisition_paths(tmp_path)
    opened: list[Path] = []
    original_open = Path.open

    def spy_open(
        self: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        if "r" in mode:
            opened.append(self.resolve())
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    private_roles = {
        paths.corpus.resolve(),
        *(path.resolve() for path in paths.views.values()),
        *(path.resolve() for path in paths.labels.values()),
    }

    # Prerequisite/envelope verification may inspect metadata for every role,
    # but it must not open any role content.
    subject.verify_acquisition_envelope(tmp_path)
    assert private_roles.isdisjoint(opened)

    # Formation opens only the corpus and both claim views.  In particular,
    # A_form gold remains unread until both action archives have sealed, and
    # A_hold/M capabilities remain physically unopened.
    subject.load_corpus_view(tmp_path)
    subject.load_block_view(tmp_path, block="A_form")
    subject.load_block_view(tmp_path, block="F_search")
    formation_allowed = {
        paths.corpus.resolve(),
        paths.views["A_form"].resolve(),
        paths.views["F_search"].resolve(),
    }
    assert private_roles.intersection(opened) == formation_allowed
    assert paths.labels["A_form"].resolve() not in opened
    assert paths.views["A_hold"].resolve() not in opened
    assert paths.labels["A_hold"].resolve() not in opened
    assert paths.views["M_search"].resolve() not in opened
    assert paths.labels["M_search"].resolve() not in opened

    opened.clear()
    subject.load_block_labels(tmp_path, block="A_form")
    assert private_roles.intersection(opened) == {paths.labels["A_form"].resolve()}

    opened.clear()
    subject.load_block_view(tmp_path, block="A_hold")
    assert private_roles.intersection(opened) == {paths.views["A_hold"].resolve()}
    assert paths.labels["A_hold"].resolve() not in opened
    assert paths.views["M_search"].resolve() not in opened
    assert paths.labels["M_search"].resolve() not in opened

    opened.clear()
    subject.load_block_labels(tmp_path, block="A_hold")
    assert private_roles.intersection(opened) == {paths.labels["A_hold"].resolve()}
    assert paths.views["M_search"].resolve() not in opened
    assert paths.labels["M_search"].resolve() not in opened

    # Only after promotion does each M role become readable, one at a time.
    opened.clear()
    subject.load_block_view(tmp_path, block="M_search")
    assert private_roles.intersection(opened) == {paths.views["M_search"].resolve()}
    assert paths.labels["M_search"].resolve() not in opened

    opened.clear()
    subject.load_block_labels(tmp_path, block="M_search")
    assert private_roles.intersection(opened) == {paths.labels["M_search"].resolve()}

    # The explicit outcome-free full verifier is still strong: it hashes every
    # role and therefore proves the final binding rather than weakening it.
    opened.clear()
    subject.verify_acquisition_receipt(tmp_path)
    assert private_roles.issubset(set(opened))
