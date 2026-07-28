from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile

import pytest

from assumption_agent.benchmarks import quac_p1_action_adapter_v1 as action
from assumption_agent.benchmarks import quac_p1_formal_acquisition_v1 as acquisition
from assumption_agent.benchmarks import quac_p1_formal_controller_v1 as controller
from assumption_agent.benchmarks import quac_p1_formal_runner_v1 as runner
from assumption_agent.benchmarks import quac_p1_runtime_v1 as runtime
from assumption_agent.benchmarks import quac_rjmc_evaluator_v1 as evaluator
from replication_runtime.quac_p1_official_v1 import contract as official_contract


@pytest.fixture
def native_tmp_path() -> Path:
    """Use the Linux filesystem so exact private modes are observable."""

    path = Path(
        tempfile.mkdtemp(
            prefix="quac-p1-runner-test-",
            dir="/home/erzhu419",
        )
    )
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _article(seed: str, family: str) -> dict[str, object]:
    tokens = [f"{seed}_evidence_{index}" for index in range(120)]
    context = " ".join(tokens)
    followup = {
        "FOLLOW": "y",
        "MAYBE_FOLLOW": "m",
        "DONT_FOLLOW": "n",
    }[family]
    qas = []
    for role, (token_index, relation) in enumerate(
        ((2, followup), (3, "y"))
    ):
        text = tokens[token_index]
        start = context.index(text)
        qas.append(
            {
                "followup": relation,
                "id": f"{seed}-qa-{role}",
                "orig_answer": {
                    "answer_start": start,
                    "text": text,
                },
                "question": f"{seed} dialogue question {role}",
            }
        )
    return {
        "paragraphs": [{"context": context, "qas": qas}],
        "section_title": f"{seed} section",
        "title": f"{seed} title",
    }


def _formal_sources() -> tuple[dict[str, object], dict[str, object]]:
    train = []
    dev = []
    for family in acquisition.FAMILY_ORDER:
        train.extend(
            _article(f"formal-train-{family}-{index}", family)
            for index in range(64)
        )
        dev.extend(
            _article(f"formal-dev-{family}-{index}", family)
            for index in range(64)
        )
    return (
        {"data": train, "version": "v0.2"},
        {"data": dev, "version": "v0.2"},
    )


def _write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def _runtime_bindings(root: Path) -> runtime.RuntimeBindings:
    python0 = root / "runtime0/bin/python"
    python1 = root / "runtime1/bin/python"
    _write(python0, b"synthetic python zero\n", 0o755)
    _write(python1, b"synthetic python one\n", 0o755)
    site0 = root / "runtime0/site"
    site1 = root / "runtime1/site"
    overlay1 = root / "runtime1/p16-site"
    base1 = root / "runtime1/base-site"
    _write(site0 / "typed.dist-info/METADATA", b"typed\n", 0o600)
    _write(site1 / "hippo.dist-info/METADATA", b"hippo\n", 0o600)
    _write(overlay1 / "torch/__init__.py", b"# torch\n", 0o600)
    _write(base1 / "distro.py", b"# distro\n", 0o600)
    minilm = root / "assets/minilm"
    llm = root / "assets/llm"
    hippo = root / "assets/hipporag"
    _write(minilm / "model.bin", b"minilm", 0o600)
    _write(llm / "model.bin", b"llm", 0o600)
    _write(hippo / "hipporag/__init__.py", b"# hippo\n", 0o600)
    return runtime.RuntimeBindings(
        gpu0_python=runtime.PythonRuntimeBinding.capture(
            executable=python0,
            import_tree=site0,
        ),
        gpu1_python=runtime.PythonRuntimeBinding.capture(
            executable=python1,
            import_tree=site1,
        ),
        gpu1_overlay_import_tree=runtime.FrozenTreeBinding.capture(
            overlay1
        ),
        gpu1_base_import_tree=runtime.FrozenTreeBinding.capture(
            base1
        ),
        minilm_asset=runtime.FrozenTreeBinding.capture(minilm),
        llm_asset=runtime.FrozenTreeBinding.capture(llm),
        hipporag_source=runtime.FrozenTreeBinding.capture(hippo),
    )


def _runtime_bindings_payload(
    bindings: runtime.RuntimeBindings,
) -> dict[str, object]:
    def python_payload(
        value: runtime.PythonRuntimeBinding,
    ) -> dict[str, object]:
        return {
            "executable": value.executable.semantic_payload(),
            "identity_sha256": value.identity_sha256,
            "import_tree": value.import_tree.semantic_payload(),
        }

    return {
        "gpu0_python": python_payload(bindings.gpu0_python),
        "gpu1_python": python_payload(bindings.gpu1_python),
        "gpu1_base_import_tree": (
            bindings.gpu1_base_import_tree.semantic_payload()
        ),
        "gpu1_overlay_import_tree": (
            bindings.gpu1_overlay_import_tree.semantic_payload()
        ),
        "hipporag_source": (
            bindings.hipporag_source.semantic_payload()
        ),
        "llm_alias": bindings.llm_alias,
        "llm_asset": bindings.llm_asset.semantic_payload(),
        "minilm_alias": bindings.minilm_alias,
        "minilm_asset": bindings.minilm_asset.semantic_payload(),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_config_payload(
    root: Path,
    bindings: runtime.RuntimeBindings,
) -> dict[str, object]:
    reconstruction_root = Path(runner.__file__).resolve().parents[2]
    effect = (
        reconstruction_root
        / "manifests/quac_p1_effect_execution_design_v1.json"
    )
    incident = (
        reconstruction_root
        / "manifests/"
        "quac_p1_postqualification_hash_only_custody_incident_v1.json"
    )
    service = root / "quac-p1-formal.service"
    _write(
        service,
        b"[Service]\nRestart=no\nExecStart=/bin/false\n",
        0o400,
    )
    modules = {
        "acquisition": acquisition,
        "action_adapter": action,
        "controller": controller,
        "evaluator": evaluator,
        "official_contract": official_contract,
        "runner": runner,
        "runtime": runtime,
    }
    body = {
        "custody_incident_file": {
            "file_sha256": _file_sha256(incident),
            "path": str(incident),
            "self_sha256": runner._CUSTODY_INCIDENT_SELF_SHA256,
        },
        "effect_design_file": {
            "file_sha256": _file_sha256(effect),
            "path": str(effect),
            "self_sha256": runner.EFFECT_DESIGN_SELF_SHA256,
        },
        "formal_root": str(root / "formal"),
        "hash_only_custody_counts": {
            "postqualification_hash_only_member_read_count": 2,
            "postqualification_hash_only_operation_count": 1,
            "semantic_decode_before_formal_count": 0,
        },
        "implementation_files": {
            name: {
                "file_sha256": _file_sha256(
                    Path(module.__file__).resolve()
                ),
                "path": str(Path(module.__file__).resolve()),
            }
            for name, module in modules.items()
        },
        "runtime_bindings": _runtime_bindings_payload(bindings),
        "schema": runner.FORMAL_CONFIG_SCHEMA,
        "service_unit": {
            "file_sha256": _file_sha256(service),
            "path": str(service),
            "unit_name": "quac-p1-formal.service",
        },
        "source_inputs": {
            role: {
                "file_sha256": runner._PINNED_SOURCE_SHA256[role],
                "path": str(root / f"{role}.json"),
                "size_bytes": runner._PINNED_SOURCE_SIZE[role],
            }
            for role in ("dev", "train")
        },
        "study_id": runner.STUDY_ID,
    }
    return {**body, "self_sha256": runner.stable_hash(body)}


def _semantic_action(
    block: runtime.RuntimeBlock,
    query: runtime.RuntimeQuery,
) -> action.ActionAdapterResult:
    seed = query.question_turns[0].question_text.split(
        " dialogue question",
        1,
    )[0]
    native = [
        document
        for document in block.documents
        if seed in document.text
    ]
    assert native
    candidate = min(
        native,
        key=lambda row: (row.context_window_ordinal, row.unit_id),
    ).unit_id
    raw = tuple(
        document.unit_id
        for document in block.documents
        if document.context_id != native[0].context_id
    )[:5]
    assert len(raw) == 5 and candidate not in raw
    graph_ids = tuple(sorted((*raw, candidate)))
    graph = evaluator.RelationalGraph(
        units=tuple(
            evaluator.EvidenceUnit(
                unit_id=unit_id,
                node_features=(
                    float(unit_id == candidate),
                    float(unit_id == candidate),
                    0.0,
                    0.0,
                ),
                dialogue_facets=(
                    int(unit_id == candidate),
                    int(unit_id == candidate),
                    0,
                    0,
                ),
            )
            for unit_id in graph_ids
        ),
        edges=(),
    )
    return action.ActionAdapterResult(
        graph=graph,
        raw_top5=graph.canonical_set(raw),
        direct_anchor_unit_ids=tuple(
            candidate for _turn in query.question_turns
        ),
        input_serialization_set_sha256=hashlib.sha256(
            query.query_id.encode("ascii")
        ).hexdigest(),
        complete_state_count=evaluator.complete_state_count(1),
    )


class _FakeBlockExecutor:
    def __init__(self, *, fail_block: str | None = None) -> None:
        self.fail_block = fail_block
        self.calls: list[tuple[str, bool]] = []
        self.blocks: dict[str, runtime.RuntimeBlock] = {}

    def __call__(
        self,
        *,
        block_name: str,
        block: runtime.RuntimeBlock,
        work_root: Path,
        official_required: bool,
    ) -> runtime.BlockRuntimeResult:
        self.calls.append((block_name, official_required))
        self.blocks[block_name] = block
        if block_name == self.fail_block:
            raise RuntimeError(f"synthetic failure in {block_name}")
        actions = {
            query.query_id: _semantic_action(block, query)
            for query in block.queries
        }
        private_payload = {
            "block_id": block.block_id,
            "rows": [
                {
                    "action": action.canonical_action_payload(
                        actions[query.query_id]
                    ),
                    "action_sha256": runtime.stable_hash(
                        action.canonical_action_payload(
                            actions[query.query_id]
                        )
                    ),
                    "query_id": query.query_id,
                }
                for query in block.queries
            ],
            "schema": runtime.ACTION_PACK_SCHEMA,
        }
        action_raw = runtime.canonical_bytes(private_payload)
        action_path = work_root / "private" / "actions.private.json"
        _write(action_path, action_raw, 0o400)
        official = (
            {
                query.query_id: actions[query.query_id].raw_top5
                for query in block.queries
            }
            if official_required
            else None
        )
        safe_body = {
            "API_or_online_evaluation_call_count": 0,
            "action_count": len(actions),
            "action_pack_file_sha256": hashlib.sha256(
                action_raw
            ).hexdigest(),
            "asset_binding_sha256": "c" * 64,
            "attempt_count": 1,
            "attempt_file_sha256": "d" * 64,
            "binding_verification_token_sha256": "b" * 64,
            "block_input_file_sha256": "e" * 64,
            "block_role": block_name,
            "corpus_count": len(block.documents),
            "index_cleanup": (
                {
                    "cleanup_verified": True,
                    "file_count": 1,
                    "total_bytes": 1,
                    "tree_sha256": "f" * 64,
                }
                if official_required
                else {
                    "cleanup_verified": True,
                    "file_count": 0,
                    "total_bytes": 0,
                    "tree_sha256": None,
                }
            ),
            "label_family_qrel_or_answer_input_count": 0,
            "logical_action_query_count": len(block.queries),
            "max_concurrent_physical_model_lanes": (
                2 if official_required else 1
            ),
            "minilm_encode_call_count": 1,
            "minilm_receipt_file_sha256": "1" * 64,
            "official_full_rankings_sha256": (
                "2" * 64 if official_required else None
            ),
            "official_index_call_count": (
                1 if official_required else 0
            ),
            "official_output_file_sha256": (
                "3" * 64 if official_required else None
            ),
            "official_required": official_required,
            "official_retrieve_call_count": (
                1 if official_required else 0
            ),
            "parallel_submission_barrier_passed": (
                True if official_required else None
            ),
            "query_count": len(block.queries),
            "retry_replay_resample_or_fallback_count": 0,
            "schema": runtime.SAFE_RESULT_SCHEMA,
            "status": "passed_label_free_block_runtime",
            "unique_embedding_count": len(block.documents),
        }
        safe_receipt = {
            **safe_body,
            "self_sha256": runtime.stable_hash(safe_body),
        }
        return runtime.BlockRuntimeResult(
            actions=actions,
            official_top5=official,
            safe_receipt=safe_receipt,
        )


class _ScientificOps:
    def __init__(self, *, promote: bool) -> None:
        self.promote = promote
        self.events: list[str] = []

    def fit_a_form(
        self,
        items,
        labels,
        *,
        block_corpus_unit_ids,
    ) -> runner.FittedEvaluator:
        self.events.append("fit")
        assert len(items) == len(labels) == 192
        assert tuple(block_corpus_unit_ids) == tuple(
            sorted(block_corpus_unit_ids)
        )
        assert {
            item.item_id for item in items
        } == {label.item_id for label in labels}
        archive = b"synthetic-frozen-model-parameters"
        return runner.FittedEvaluator(
            model=object(),
            parameter_sha256=hashlib.sha256(archive).hexdigest(),
            parameter_archive=archive,
        )

    def select_measurement(
        self,
        *,
        block,
        items,
        fitted,
        hipporag_top5,
        block_corpus_unit_ids,
    ) -> runner.MeasurementActions:
        del fitted
        self.events.append(f"select:{block}")
        rows = []
        for item in sorted(items, key=lambda row: row.item_id):
            candidates = tuple(
                unit_id
                for unit_id in item.graph.unit_ids
                if unit_id not in item.raw_top5
            )
            assert len(candidates) == 1
            e1 = (
                item.graph.canonical_set(
                    (*item.raw_top5[:-1], candidates[0])
                )
                if self.promote
                else item.raw_top5
            )
            rows.append(
                controller.ActionRow(
                    item_id=item.item_id,
                    E0=item.raw_top5,
                    E1=e1,
                    RAW=item.raw_top5,
                    official_HippoRAG=tuple(
                        hipporag_top5[item.item_id]
                    ),
                )
            )
        native = controller.SealedStageActions(
            block=block,
            corpus_unit_ids_sha256=controller.stable_hash(
                list(block_corpus_unit_ids)
            ),
            rows=tuple(rows),
        )
        payload = native.payload()
        return runner.MeasurementActions(
            block=block,
            native=native,
            corpus_unit_ids=tuple(block_corpus_unit_ids),
            payload=payload,
            action_sha256=runner.stable_hash(payload),
        )

    def score_measurement(
        self,
        actions,
        labels,
    ) -> runner.MeasurementScore:
        self.events.append(f"score:{actions.block}")
        native = controller.score_sealed_stage(
            actions.native,
            labels,
            block_corpus_unit_ids=actions.corpus_unit_ids,
        )
        comparison = native.comparison("E0")
        return runner.MeasurementScore(
            block=actions.block,
            native=native,
            safe_payload=native.safe_payload(),
            e1_minus_e0=comparison.net,
            p_numerator=comparison.exact.numerator,
            p_denominator=comparison.exact.denominator,
            promoted=(
                native.promotion
                if actions.block == "A_hold"
                else native.l5
            ),
        )

    def safe_terminal(
        self,
        *,
        a_hold,
        m_search,
        fitted,
        action_commitments,
        runtime_commitments,
        m_materialization_count_before_promotion,
    ):
        self.events.append("terminal")
        return controller.safe_terminal(
            a_hold=a_hold.native,
            m_search=(
                None if m_search is None else m_search.native
            ),
            model_parameter_sha256=fitted.parameter_sha256,
            action_commitments=action_commitments,
            runtime_commitments=runtime_commitments,
            M_materialization_count_before_promotion=(
                m_materialization_count_before_promotion
            ),
        )


@dataclass
class _SecretFactory:
    value: bytes
    calls: int = 0

    def __call__(self, size: int) -> bytes:
        assert size == 32
        self.calls += 1
        return self.value


def test_promoted_lifecycle_runs_exact_order_and_opens_M_once(
    native_tmp_path: Path,
) -> None:
    train, dev = _formal_sources()
    executor = _FakeBlockExecutor()
    science = _ScientificOps(promote=True)
    secret = _SecretFactory(b"\x61" * 32)
    root = (native_tmp_path / "formal").absolute()
    result = runner.run_formal_once(
        train_obj=train,
        dev_obj=dev,
        work_root=root,
        block_executor=executor,
        scientific_ops=science,
        secret_factory=secret,
    )
    assert secret.calls == 1
    assert executor.calls == [
        ("A_form", False),
        ("A_hold", True),
        ("M_search", True),
    ]
    assert science.events == [
        "fit",
        "select:A_hold",
        "score:A_hold",
        "select:M_search",
        "score:M_search",
        "terminal",
    ]
    assert result.terminal["status"] == (
        "VALID_COMPLETE_PROMOTED_M_MEASURED"
    )
    assert result.terminal["block_execution_counts"] == {
        "A_form": 1,
        "A_hold": 1,
        "M_search": 1,
    }
    assert result.terminal["effect_design_self_sha256"] == (
        runner.EFFECT_DESIGN_SELF_SHA256
    )
    assert stat.S_IMODE(
        (root / runner.SECRET_FILENAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(result.terminal_path.stat().st_mode) == 0o400
    assert (root / "private/M_search.capability.private.json").is_file()
    assert (root / "safe/M_search.score.safe.json").is_file()

    # The bridge is recent-first and contains no query-to-native-context key.
    a_hold = executor.blocks["A_hold"]
    assert a_hold.queries[0].question_turns[0].question_text.endswith(
        "dialogue question 1"
    )
    block_payload = runtime.block_payload(a_hold)
    serialized = json.dumps(block_payload, sort_keys=True)
    for forbidden in (
        '"family"',
        '"split"',
        '"qrel"',
        '"answer"',
        "native_context_id",
    ):
        assert forbidden not in serialized


def test_nonpromotion_is_valid_and_M_remains_only_opaque_reservation(
    native_tmp_path: Path,
) -> None:
    train, dev = _formal_sources()
    executor = _FakeBlockExecutor()
    root = (native_tmp_path / "formal").absolute()
    result = runner.run_formal_once(
        train_obj=train,
        dev_obj=dev,
        work_root=root,
        block_executor=executor,
        scientific_ops=_ScientificOps(promote=False),
        secret_factory=_SecretFactory(b"\x62" * 32),
    )
    assert executor.calls == [
        ("A_form", False),
        ("A_hold", True),
    ]
    assert result.terminal["status"] == (
        "VALID_NONPROMOTION_M_UNOPENED"
    )
    assert result.terminal["block_execution_counts"]["M_search"] == 0
    reservation = json.loads(
        (root / "safe/M_search.reservation.safe.json").read_text(
            "ascii"
        )
    )
    assert reservation["materialization_count"] == 0
    assert reservation["materialized_path_count"] == 0
    assert "rows" not in reservation
    assert not (root / "stages/M_search").exists()
    assert not (
        root / "private/M_search.capability.private.json"
    ).exists()


def test_failure_is_safe_terminal_and_attempt_cannot_retry(
    native_tmp_path: Path,
) -> None:
    train, dev = _formal_sources()
    executor = _FakeBlockExecutor(fail_block="A_hold")
    secret = _SecretFactory(b"\x63" * 32)
    root = (native_tmp_path / "formal").absolute()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_formal_once(
            train_obj=train,
            dev_obj=dev,
            work_root=root,
            block_executor=executor,
            scientific_ops=_ScientificOps(promote=True),
            secret_factory=secret,
        )
    failure = json.loads(
        (root / runner.FAILURE_FILENAME).read_text("ascii")
    )
    assert failure["status"] == (
        "implementation_or_infrastructure_invalid"
    )
    assert failure["stage"] == (
        "A_hold_label_free_three_arm_runtime"
    )
    assert failure["API_or_online_evaluation_call_count"] == 0
    assert failure[
        "retry_replay_resample_repair_or_fallback_authorized"
    ] is False
    assert "synthetic failure" not in json.dumps(failure)
    assert secret.calls == 1

    with pytest.raises(
        runner.QuacP1FormalRunnerError,
        match="retry is forbidden",
    ):
        runner.run_formal_once(
            train_obj=train,
            dev_obj=dev,
            work_root=root,
            block_executor=executor,
            scientific_ops=_ScientificOps(promote=True),
            secret_factory=secret,
        )
    assert secret.calls == 1
    assert executor.calls == [
        ("A_form", False),
        ("A_hold", True),
    ]


def test_bad_secret_consumes_attempt_and_never_reaches_acquisition(
    native_tmp_path: Path,
) -> None:
    train, dev = _formal_sources()
    called = []

    def acquisition_factory(_train, _dev, _secret):
        called.append(True)
        raise AssertionError("must not acquire")

    root = (native_tmp_path / "formal").absolute()
    with pytest.raises(
        runner.QuacP1FormalRunnerError,
        match="exactly 32 bytes",
    ):
        runner.run_formal_once(
            train_obj=train,
            dev_obj=dev,
            work_root=root,
            block_executor=_FakeBlockExecutor(),
            scientific_ops=_ScientificOps(promote=True),
            acquisition_factory=acquisition_factory,
            secret_factory=lambda _size: b"bad",
        )
    assert not called
    assert (root / runner.ATTEMPT_FILENAME).is_file()
    assert (root / runner.FAILURE_FILENAME).is_file()
    assert not (root / runner.SECRET_FILENAME).exists()


def test_bound_runtime_executor_uses_one_verified_token_and_current_api(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _runtime_bindings(native_tmp_path / "bindings")
    verified = runtime.verify_runtime_bindings_once(
        bindings,
        source_access_count=0,
    )
    block = runtime.RuntimeBlock(
        block_id="1" * 64,
        documents=tuple(
            action.BlockDocument(
                unit_id=f"{index + 10:064x}",
                context_id=f"{index + 100:064x}",
                title="title",
                section_title="section",
                context_window_ordinal=0,
                text=f"document {index}",
            )
            for index in range(5)
        ),
        queries=(
            runtime.RuntimeQuery(
                query_id="2" * 64,
                question_turns=(
                    action.QuestionTurn("question"),
                ),
            ),
        ),
    )
    observed = []
    expected = runtime.BlockRuntimeResult(
        actions={},
        official_top5=None,
        safe_receipt={},
    )

    def fake_run_block(**kwargs):
        observed.append(kwargs)
        return expected

    monkeypatch.setattr(runtime, "run_block", fake_run_block)
    executor = runner.BoundRuntimeExecutor(
        bindings=bindings,
        verified_bindings=verified,
        encoder=object(),
        official_lane=object(),
    )
    result = executor(
        block_name="A_form",
        block=block,
        work_root=native_tmp_path / "block",
        official_required=False,
    )
    assert result is expected
    assert len(observed) == 1
    assert observed[0]["block_role"] == "A_form"
    assert observed[0]["verified_bindings"] is verified
    assert observed[0]["bindings"] is bindings
    assert observed[0]["official_lane"] is None
    assert "official_required" not in observed[0]


def test_scientific_core_cannot_be_used_as_a_production_cli() -> None:
    with pytest.raises(
        runner.QuacP1FormalRunnerError,
        match="scientific core without a CLI",
    ):
        runner.main([])
