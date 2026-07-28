from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import pytest

from assumption_agent.benchmarks import (
    quac_p1_formal_acquisition_v1 as subject,
)
from assumption_agent.benchmarks import (
    quac_p1_formal_controller_v1 as controller,
)


SMALL_QUOTAS = {"A_form": 1, "A_hold": 1, "M_search": 1}


def _article(
    seed: str,
    *,
    families: tuple[str, ...] = ("FOLLOW", "MAYBE_FOLLOW", "DONT_FOLLOW"),
    title: str | None = None,
    context: str | None = None,
) -> dict[str, object]:
    if context is None:
        tokens = [f"{seed}_evidence_{index}" for index in range(120)]
        context = " ".join(tokens)
    else:
        tokens = context.split()
        assert len(tokens) >= 20

    followup_for_family = {
        "FOLLOW": "y",
        "MAYBE_FOLLOW": "m",
        "DONT_FOLLOW": "n",
    }
    # Each desired family gets its own previous/current pair.  The two answer
    # tokens are deliberately in canonical window zero, so one unit can
    # satisfy both distinct qrel roles.
    qas: list[dict[str, object]] = []
    for pair_index, family in enumerate(families):
        prior_token = tokens[2 + pair_index * 4]
        current_token = tokens[3 + pair_index * 4]
        for role_index, (token, followup) in enumerate(
            (
                (prior_token, followup_for_family[family]),
                (current_token, "y"),
            )
        ):
            start = context.index(token)
            qas.append(
                {
                    "answers": [],
                    "followup": followup,
                    "id": f"{seed}-qa-{pair_index}-{role_index}",
                    "orig_answer": {
                        "answer_end": start + len(token),
                        "answer_start": start,
                        "text": token,
                    },
                    "question": (
                        f"{seed} dialogue question {pair_index} "
                        f"{role_index}"
                    ),
                    "yesno": "x",
                }
            )
    return {
        "background": f"{seed} background",
        "paragraphs": [{"context": context, "qas": qas}],
        "section_title": f"{seed} section",
        "title": title if title is not None else f"{seed} title",
    }


def _payload(articles: list[dict[str, object]]) -> dict[str, object]:
    return {"data": articles, "version": "v0.2"}


def _small_fixture() -> tuple[dict[str, object], dict[str, object]]:
    return (
        _payload([_article(f"train-{index}") for index in range(3)]),
        _payload([_article(f"dev-{index}") for index in range(6)]),
    )


def _formal_fixture() -> tuple[dict[str, object], dict[str, object]]:
    train: list[dict[str, object]] = []
    dev: list[dict[str, object]] = []
    for family in subject.FAMILY_ORDER:
        train.extend(
            _article(
                f"formal-train-{family}-{index}",
                families=(family,),
            )
            for index in range(64)
        )
        # 32 A_hold + 32 M_search components for each family.
        dev.extend(
            _article(
                f"formal-dev-{family}-{index}",
                families=(family,),
            )
            for index in range(64)
        )
    return _payload(train), _payload(dev)


def _promotion(
    broker: subject.TrustedAcquisitionBroker,
    *,
    promoted: bool = True,
) -> subject.PromotionProof:
    return subject.PromotionProof.create(
        selection_commitment=broker.selection_commitment,
        a_hold_score_receipt_sha256="a" * 64,
        aggregate_e1_minus_e0=1,
        p_numerator=1,
        p_denominator=10,
        promoted=promoted,
    )


@pytest.fixture
def native_tmp_path() -> Path:
    """Use the Linux filesystem so exact mode-0400 checks are meaningful."""

    path = Path(
        tempfile.mkdtemp(
            prefix="quac-p1-acquisition-test-",
            dir="/home/erzhu419",
        )
    )
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_barrier(
    root: Path,
    name: str,
    payload: dict[str, object],
) -> Path:
    path = root / name
    path.write_bytes(subject.canonical_bytes(payload))
    path.chmod(0o400)
    return path


def _a_form_action_payload(
    broker: subject.TrustedAcquisitionBroker,
) -> dict[str, object]:
    material = broker.runtime_material("A_form")
    unit_ids = [row.unit_id for row in material.documents[:5]]
    action = {
        "complete_state_count": 1,
        "direct_anchor_unit_ids": [unit_ids[0]],
        "graph": {
            "edges": [],
            "units": [
                {
                    "dialogue_facets": [0, 0, 0, 0],
                    "node_features_micro": [0, 0, 0, 0],
                    "unit_id": unit_id,
                }
                for unit_id in unit_ids
            ],
        },
        "input_serialization_set_sha256": "a" * 64,
        "raw_top5": unit_ids,
        "schema": "quac_p1_action_adapter_v1",
        "version": "quac_p1_action_adapter_v1",
    }
    return {
        "block_id": material.block_id,
        "rows": [
            {
                "action": action,
                "action_sha256": subject.stable_hash(action),
                "query_id": query.query_id,
            }
            for query in material.queries
        ],
        "schema": "quac_p1_runtime_v1_private_action_pack_v1",
    }


def _register_a_form_barrier(
    broker: subject.TrustedAcquisitionBroker,
    root: Path,
) -> subject.LateLabelCapability:
    payload = _a_form_action_payload(broker)
    return broker.register_durable_action_barrier(
        block="A_form",
        action_path=_write_barrier(
            root,
            "A_form.actions.private.json",
            payload,
        ),
        expected_payload=payload,
    )


def _complete_a_form_and_register_model_seal(
    broker: subject.TrustedAcquisitionBroker,
    root: Path,
) -> subject.LateLabelCapability:
    label_capability = _register_a_form_barrier(broker, root)
    broker.open_late_labels(label_capability)
    seal = broker.issue_a_form_model_seal(
        model_parameter_sha256="f" * 64,
    )
    seal_path = _write_barrier(
        root,
        "A_form.model_seal.private.json",
        seal.payload(),
    )
    broker.register_durable_a_form_model_seal(
        seal=seal,
        seal_path=seal_path,
    )
    return label_capability


def _prepare_promoted_a_hold(
    broker: subject.TrustedAcquisitionBroker,
    root: Path,
    *,
    complete_a_form: bool = True,
    authorize: bool = True,
) -> tuple[
    subject.MSearchCapability | None,
    controller.StageScore,
    Path,
]:
    if complete_a_form:
        _complete_a_form_and_register_model_seal(broker, root)
    material = broker.runtime_material("A_hold")
    corpus_ids = tuple(row.unit_id for row in material.documents)
    actions: list[controller.ActionRow] = []
    for query in material.queries:
        seed = query.question_turns[0].split(
            " dialogue question",
            maxsplit=1,
        )[0]
        native = next(
            row.unit_id
            for row in material.documents
            if (
                row.title == f"{seed} title"
                and row.context_window_ordinal == 0
            )
        )
        raw = tuple(
            unit_id
            for unit_id in corpus_ids
            if unit_id != native
        )[:5]
        actions.append(
            controller.ActionRow(
                item_id=query.query_id,
                E0=raw,
                E1=(native, *raw[:4]),
                RAW=raw,
                official_HippoRAG=raw,
            )
        )
    sealed = controller.SealedStageActions(
        block="A_hold",
        corpus_unit_ids_sha256=controller.stable_hash(
            list(corpus_ids)
        ),
        rows=tuple(sorted(actions, key=lambda row: row.item_id)),
    )
    action_payload = sealed.payload()
    late_capability = broker.register_durable_action_barrier(
        block="A_hold",
        action_path=_write_barrier(
            root,
            "A_hold.actions.private.json",
            action_payload,
        ),
        expected_payload=action_payload,
    )
    labels = broker.open_late_labels(late_capability)
    controller_labels = tuple(
        controller.LateLabelRow(
            item_id=row.work_id.removeprefix("quac-work-v1-"),
            family=row.family,
            previous_qrel=row.previous_turn_orig_answer,
            current_qrel=row.current_turn_orig_answer,
        )
        for row in labels.rows
    )
    stage_score = controller.score_sealed_stage(
        sealed,
        controller_labels,
        block_corpus_unit_ids=corpus_ids,
    )
    assert stage_score.promotion is True
    score_path = _write_barrier(
        root,
        "A_hold.score.safe.json",
        stage_score.safe_payload(),
    )
    capability = (
        broker.authorize_m_search_from_stage_score(
            stage_score=stage_score,
            score_receipt_path=score_path,
        )
        if authorize
        else None
    )
    return capability, stage_score, score_path


def _all_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        result.update(value)
        for child in value.values():
            result.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_all_keys(child))
    return result


def test_global_component_is_exact_title_or_context_transitive_union() -> None:
    shared_context = " ".join(f"shared_{index}" for index in range(120))
    train = _payload(
        [
            _article(
                "left",
                title="same exact title",
            )
        ]
    )
    dev = _payload(
        [
            _article(
                "bridge",
                title="same exact title",
                context=shared_context,
            ),
            _article(
                "right",
                title="different exact title",
                context=shared_context,
            ),
        ]
    )
    index = subject.build_source_index(train, dev)
    assert index.paragraph_count == 3
    assert index.component_count == 1
    assert len({row.component_commitment for row in index.items}) == 1

    # Case and whitespace are exact; they are not normalized before union.
    exact = subject.build_source_index(
        _payload([_article("a", title="Page")]),
        _payload([_article("b", title="page")]),
    )
    assert exact.component_count == 2


def test_required_subset_and_singular_orig_answer_match_qualifier() -> None:
    train, dev = _small_fixture()
    train["data"][0]["extra_article_field"] = {"accepted": True}
    train["data"][0]["paragraphs"][0]["extra_paragraph_field"] = 1
    index = subject.build_source_index(train, dev)
    assert index.items

    broken = deepcopy(train)
    del broken["data"][0]["paragraphs"][0]["qas"][0]["orig_answer"]
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="required fields",
    ):
        subject.build_source_index(broken, dev)

    # The plural annotator answer list is never a fallback qrel source.
    cannot = deepcopy(train)
    qa = cannot["data"][0]["paragraphs"][0]["qas"][0]
    qa["orig_answer"] = {
        "answer_start": 0,
        "text": subject.CANNOTANSWER,
    }
    qa["answers"] = [
        {
            "answer_start": 0,
            "text": cannot["data"][0]["paragraphs"][0]["context"].split()[0],
        }
    ]
    reduced = subject.build_source_index(cannot, dev)
    assert len(reduced.items) < len(index.items)


def test_one_global_residual_flow_recovers_from_a_greedy_dead_end() -> None:
    # TRAIN component 0 is flexible FOLLOW/MAYBE; component 1 is FOLLOW-only.
    # A one-pass greedy assignment can consume FOLLOW with component 0 and
    # strand component 1.  The residual flow must reverse that early edge.
    train = _payload(
        [
            _article(
                "flexible",
                families=("FOLLOW", "MAYBE_FOLLOW"),
            ),
            _article("follow-only", families=("FOLLOW",)),
            _article("dont-only", families=("DONT_FOLLOW",)),
        ]
    )
    dev = _payload(
        [
            _article(f"dev-{index}")
            for index in range(6)
        ]
    )
    index = subject.build_source_index(train, dev)
    flexible = next(
        row.component_commitment
        for row in index.items
        if row.question_text.startswith("flexible")
    )
    follow_only = next(
        row.component_commitment
        for row in index.items
        if row.question_text.startswith("follow-only")
    )

    chosen: subject.SelectionSecret | None = None
    for value in range(10_000):
        candidate = subject.SelectionSecret(value.to_bytes(32, "big"))
        component_order = sorted(
            (flexible, follow_only),
            key=lambda component: (
                candidate.digest(
                    "component-order-v1",
                    {"component_commitment": component},
                ),
                component,
            ),
        )
        flexible_slots = sorted(
            (
                ("A_form", "FOLLOW"),
                ("A_form", "MAYBE_FOLLOW"),
            ),
            key=lambda slot: (
                candidate.digest(
                    "component-slot-order-v1",
                    {
                        "block": slot[0],
                        "component_commitment": flexible,
                        "family": slot[1],
                    },
                ),
                subject.BLOCK_ORDER.index(slot[0]),
                subject.FAMILY_ORDER.index(slot[1]),
            ),
        )
        if (
            component_order[0] == flexible
            and flexible_slots[0] == ("A_form", "FOLLOW")
        ):
            chosen = candidate
            break
    assert chosen is not None

    plan = subject.select_study(index, chosen, quotas=SMALL_QUOTAS)
    assert not hasattr(plan, "source_index")
    a_form = plan.rows("A_form")
    assert Counter(row.family for row in a_form) == {
        family: 1 for family in subject.FAMILY_ORDER
    }
    flexible_row = next(
        row for row in a_form if row.source.component_commitment == flexible
    )
    assert flexible_row.family == "MAYBE_FOLLOW"
    assert len(
        {
            row.source.component_commitment
            for row in plan.selected
        }
    ) == 9


def test_label_free_view_and_late_two_role_qrels_are_separate(
    native_tmp_path: Path,
) -> None:
    train, dev = _small_fixture()
    broker = subject.acquire_decoded_sources_once(
        train,
        dev,
        b"\x17" * 32,
        quotas=SMALL_QUOTAS,
    )
    view = broker.view_pack("A_form")
    payload = view.payload()
    assert payload["block"] == "A_form"
    assert set(payload["rows"][0]) == {
        "query_text",
        "recent_questions",
        "work_id",
    }
    assert not (_all_keys(payload) & subject.FORBIDDEN_VIEW_KEYS)

    serialized = view.canonical_bytes()
    assert b"private" not in serialized
    assert subject.decode_strict_pack(
        serialized,
        expected_schema=subject.VIEW_SCHEMA,
    ) == payload

    # Labels are unavailable through a view and open only after an action seal
    # has been cryptographically bound into a one-use late-label capability.
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="arbitrary action-seal",
    ):
        broker.issue_late_label_capability(
            block="A_form",
            action_seal_sha256="b" * 64,
        )
    token = _register_a_form_barrier(broker, native_tmp_path)
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="exact A_form model seal",
    ):
        broker.runtime_material("A_hold")
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="exact A_form model seal",
    ):
        broker.view_pack("A_hold")
    labels = broker.open_late_labels(token)
    assert labels.block == "A_form"
    assert len(labels.rows) == 3
    assert any(
        set(row.previous_turn_orig_answer).intersection(
            row.current_turn_orig_answer
        )
        for row in labels.rows
    )
    assert subject.decode_strict_pack(
        labels.canonical_bytes(),
        expected_schema=subject.LABEL_SCHEMA,
    ) == labels.payload()
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="replay",
    ):
        broker.open_late_labels(token)
    seal = broker.issue_a_form_model_seal(
        model_parameter_sha256="d" * 64,
    )
    seal_path = _write_barrier(
        native_tmp_path,
        "A_form.model_seal.private.json",
        seal.payload(),
    )

    class ForgedModelSeal(subject.AFormModelSeal):
        pass

    forged_seal = ForgedModelSeal(
        selection_commitment=seal.selection_commitment,
        action_seal_sha256=seal.action_seal_sha256,
        label_pack_sha256=seal.label_pack_sha256,
        model_parameter_sha256=seal.model_parameter_sha256,
        seal_mac=seal.seal_mac,
    )
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="type is not exact",
    ):
        broker.register_durable_a_form_model_seal(
            seal=forged_seal,
            seal_path=seal_path,
        )
    broker.register_durable_a_form_model_seal(
        seal=seal,
        seal_path=seal_path,
    )
    assert len(broker.runtime_material("A_hold").queries) == 3
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="already been registered",
    ):
        broker.register_durable_a_form_model_seal(
            seal=seal,
            seal_path=seal_path,
        )


def test_durable_barrier_rejects_symlink_and_detects_postseal_drift(
    native_tmp_path: Path,
) -> None:
    train, dev = _small_fixture()
    broker = subject.acquire_decoded_sources_once(
        train,
        dev,
        b"\x19" * 32,
        quotas=SMALL_QUOTAS,
    )
    payload = _a_form_action_payload(broker)
    target = _write_barrier(
        native_tmp_path,
        "direct.actions.private.json",
        payload,
    )
    symlink = native_tmp_path / "linked.actions.private.json"
    symlink.symlink_to(target)
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="unavailable",
    ):
        broker.register_durable_action_barrier(
            block="A_form",
            action_path=symlink,
            expected_payload=payload,
        )
    capability = broker.register_durable_action_barrier(
        block="A_form",
        action_path=target,
        expected_payload=payload,
    )
    target.chmod(0o600)
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="changed",
    ):
        broker.open_late_labels(capability)


def test_m_search_is_opaque_until_one_valid_nonreplayable_token(
    native_tmp_path: Path,
) -> None:
    train, dev = _formal_fixture()
    broker = subject.acquire_decoded_sources_once(
        train,
        dev,
        b"\x23" * 32,
    )
    reservation = broker.m_reservation_receipt()
    assert reservation["materialization_count"] == 0
    assert reservation["materialized_path_count"] == 0
    assert set(reservation) == {
        "block",
        "item_count",
        "materialization_count",
        "materialized_path_count",
        "opaque_reservation_commitment",
        "schema",
        "selection_commitment",
        "study_id",
    }
    assert "rows" not in reservation
    assert "work_id" not in subject.canonical_bytes(reservation).decode()
    subject.decode_strict_pack(
        subject.canonical_bytes(reservation),
        expected_schema=subject.RESERVATION_SCHEMA,
    )
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="sealed",
    ):
        broker.private_rows("M_search")
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="not materialized",
    ):
        broker.view_pack("M_search")
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="external promotion",
    ):
        broker.issue_m_search_capability(
            _promotion(broker, promoted=False)
        )

    capability, stage_score, score_path = _prepare_promoted_a_hold(
        broker,
        native_tmp_path,
        authorize=False,
    )
    assert capability is None

    class ForgedStageScore(controller.StageScore):
        pass

    forged_stage_score = ForgedStageScore(**stage_score.__dict__)
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="actual promoted controller StageScore",
    ):
        broker.authorize_m_search_from_stage_score(
            stage_score=forged_stage_score,
            score_receipt_path=score_path,
        )
    capability = broker.authorize_m_search_from_stage_score(
        stage_score=stage_score,
        score_receipt_path=score_path,
    )
    forged = subject.MSearchCapability(
        reservation_commitment=capability.reservation_commitment,
        selection_commitment=capability.selection_commitment,
        promotion_proof_sha256=capability.promotion_proof_sha256,
        capability_mac="0" * 64,
    )
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="forged",
    ):
        broker.materialize_m_search(forged)
    assert broker.m_reservation_receipt()["materialization_count"] == 0

    materialized = broker.materialize_m_search(capability)
    assert materialized.view_pack.block == "M_search"
    assert not hasattr(materialized, "private_rows")
    assert len(broker.runtime_material("M_search").queries) == 96
    assert broker.m_reservation_receipt()["materialization_count"] == 1
    assert broker.m_reservation_receipt()["materialized_path_count"] == 0
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="replay",
    ):
        broker.materialize_m_search(capability)
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="arbitrary",
    ):
        broker.record_m_search_materialized_paths_once(2)
    registry = broker.m_search_materialized_registry_payload()
    broker.register_durable_m_search_materialized_registry(
        registry_path=_write_barrier(
            native_tmp_path,
            "M_search.materialized_registry.private.json",
            registry,
        ),
        expected_payload=registry,
    )
    assert broker.m_reservation_receipt()["materialized_path_count"] == 1
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="replayed",
    ):
        broker.register_durable_m_search_materialized_registry(
            registry_path=(
                native_tmp_path
                / "M_search.materialized_registry.private.json"
            ),
            expected_payload=registry,
        )


def test_formal_a_form_hmac_folds_are_exactly_balanced_and_stable() -> None:
    train, dev = _formal_fixture()
    broker = subject.acquire_decoded_sources_once(
        train,
        dev,
        b"\x31" * 32,
    )
    first = broker.a_form_folds()
    second = broker.a_form_folds()
    assert first == second
    assert len(first) == 192
    assert tuple(
        sum(fold == index for fold in first.values())
        for index in range(5)
    ) == (39, 39, 38, 38, 38)

    other = subject.acquire_decoded_sources_once(
        train,
        dev,
        b"\x32" * 32,
    ).a_form_folds()
    assert other != first


def test_strict_pack_rejects_noncanonical_and_forbidden_view_fields() -> None:
    train, dev = _small_fixture()
    broker = subject.acquire_decoded_sources_once(
        train,
        dev,
        b"\x41" * 32,
        quotas=SMALL_QUOTAS,
    )
    payload = broker.view_pack("A_form").payload()
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="canonical",
    ):
        subject.decode_strict_pack(
            subject.canonical_bytes(payload) + b"\n",
            expected_schema=subject.VIEW_SCHEMA,
        )

    leaked = deepcopy(payload)
    leaked["rows"][0]["family"] = "FOLLOW"
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="row key set|forbidden",
    ):
        subject.decode_strict_pack(
            subject.canonical_bytes(leaked),
            expected_schema=subject.VIEW_SCHEMA,
        )

    duplicate = (
        b'{"block":"A_form","block":"A_hold","rows":[],"schema":"'
        + subject.VIEW_SCHEMA.encode("ascii")
        + b'","selection_commitment":"'
        + b"0" * 64
        + b'","study_id":"'
        + subject.STUDY_ID.encode("ascii")
        + b'"}'
    )
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="duplicate",
    ):
        subject.decode_strict_pack(
            duplicate,
            expected_schema=subject.VIEW_SCHEMA,
        )


def test_secret_width_and_capability_issuance_are_one_epoch_only(
    native_tmp_path: Path,
) -> None:
    train, dev = _formal_fixture()
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="exactly 32 bytes",
    ):
        subject.acquire_decoded_sources_once(
            train,
            dev,
            b"short",
        )

    broker = subject.acquire_decoded_sources_once(
        train,
        dev,
        b"\x51" * 32,
    )
    label = _register_a_form_barrier(broker, native_tmp_path)
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="already been issued",
    ):
        payload = _a_form_action_payload(broker)
        broker.register_durable_action_barrier(
            block="A_form",
            action_path=_write_barrier(
                native_tmp_path,
                "A_form.replay.actions.private.json",
                payload,
            ),
            expected_payload=payload,
        )
    forged = subject.LateLabelCapability(
        block=label.block,
        selection_commitment=label.selection_commitment,
        action_seal_sha256=label.action_seal_sha256,
        capability_mac="0" * 64,
    )
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="forged",
    ):
        broker.open_late_labels(forged)
    broker.open_late_labels(label)
    seal = broker.issue_a_form_model_seal(
        model_parameter_sha256="e" * 64,
    )
    broker.register_durable_a_form_model_seal(
        seal=seal,
        seal_path=_write_barrier(
            native_tmp_path,
            "A_form.model_seal.private.json",
            seal.payload(),
        ),
    )
    capability, stage_score, score_path = _prepare_promoted_a_hold(
        broker,
        native_tmp_path,
        complete_a_form=False,
    )
    assert capability is not None
    with pytest.raises(
        subject.QuacP1FormalAcquisitionError,
        match="already been issued",
    ):
        broker.authorize_m_search_from_stage_score(
            stage_score=stage_score,
            score_receipt_path=score_path,
        )
    assert subject.decode_strict_pack(
        subject.canonical_bytes(capability.payload()),
        expected_schema=subject.M_CAPABILITY_SCHEMA,
    ) == capability.payload()
