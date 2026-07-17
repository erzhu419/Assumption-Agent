from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import threading
from typing import Mapping, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import synthetic_typed_graph_causal_acquisition_v1 as acq
from assumption_agent.benchmarks import synthetic_typed_graph_causal_grammar_v1 as grammar
from assumption_agent.benchmarks import synthetic_typed_graph_causal_runner_v1 as runner


# Public unit-test fixture only.  Formal custody code is never called here.
TEST_SEED = bytes(range(32))


class HashEncoder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.asarray(
                [float(digest[index % len(digest)] + 1) for index in range(384)],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


class FakeOfficialRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.live = 0
        self.max_live = 0

    @property
    def safe_binding(self) -> Mapping[str, object]:
        return {"runtime": "offline_fake_test_only", "revision": 1}

    def retrieve(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]:
        assert question
        assert len(paragraphs) == 32
        with self._lock:
            self.calls += 1
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        try:
            # Stable label-free incumbent.  The fake writes nothing and has no
            # network or model behavior; it is only an interface fixture.
            return (0, 1, 2, 3, 4)
        finally:
            with self._lock:
                self.live -= 1

    def fresh_reverify(self) -> Mapping[str, object]:
        return dict(self.safe_binding)


@pytest.fixture(scope="module")
def private_packs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Path | None]]:
    root = tmp_path_factory.mktemp("synthetic-private-packs")
    blocks = grammar.generate_all_blocks(TEST_SEED)
    derangement = dict(
        grammar.evaluator_label_derangement(blocks["A_form"], seed=TEST_SEED)
    )
    result: dict[str, tuple[Path, Path | None]] = {}
    for block in grammar.BLOCK_ORDER:
        items = blocks[block]
        view_rows = [acq._view_row(item) for item in items]
        view_ids = {
            item.item_commitment_sha256: str(row["opaque_view_sha256"])
            for item, row in zip(items, view_rows)
        }
        view = acq._pack(acq.VIEW_SCHEMA, block, view_rows)
        view_path = root / f"{block}.view.json"
        acq._write_json_exclusive(view_path, view, acq.PRIVATE_MODE)
        label_path: Path | None = None
        if block != "F_search":
            labels = acq._pack(
                acq.LABEL_SCHEMA,
                block,
                acq._label_rows(
                    block,
                    items,
                    view_ids=view_ids,
                    derangement=derangement if block == "A_form" else None,
                ),
            )
            label_path = root / f"{block}.labels.json"
            acq._write_json_exclusive(label_path, labels, acq.PRIVATE_MODE)
        result[block] = (view_path, label_path)
    return result


def test_preseed_amendment_and_design_are_self_hashed_and_bind_new_semantics() -> None:
    project_root = Path(__file__).resolve().parents[1]
    design = acq.verify_frozen_design(project_root)
    assert design["design_sha256"] == acq.DESIGN_SHA256
    assert "design_based_randomization_p_value" in design["claim_boundary"]["seed_level_inference"]
    negative_kinds = {
        row["negative_kind"]
        for row in design["family_registry"]
        if row["family_role"] == grammar.TRAIN_NEGATIVE_2
    }
    assert negative_kinds == {
        "edge_present_but_query_and_gold_are_independent_direct_cue"
    }
    assert not any("decoy" in value for value in negative_kinds)


def test_external_freeze_verifies_nested_project_against_actual_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    project = repo / "reconstruction_v2"
    project.mkdir(parents=True)
    bindings = []
    for relative in sorted(acq.REQUIRED_FREEZE_PATHS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"bound:{relative}\n".encode()
        path.write_bytes(raw)
        bindings.append(
            {
                "relative_path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob_sha1": acq._git_blob_sha1(raw),
            }
        )
    body = {
        "schema": acq.IMPLEMENTATION_FREEZE_SCHEMA,
        "design_sha256": acq.DESIGN_SHA256,
        "amendment_sha256": acq.AMENDMENT_SHA256,
        "formal_seed_or_cohort_exists": False,
        "bindings": bindings,
    }
    freeze = {**body, "implementation_freeze_sha256": acq.semantic_hash(body)}
    freeze_path = project / acq.IMPLEMENTATION_FREEZE_RELATIVE_PATH
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_bytes(acq.canonical_bytes(freeze) + b"\n")
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.invalid"), cwd=repo, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=repo, check=True)
    subprocess.run(("git", "add", "reconstruction_v2"), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-qm", "freeze"), cwd=repo, check=True)
    observed, head = acq.verify_implementation_freeze(project)
    assert observed == freeze
    assert head == subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repo).decode().strip()


def test_seed_custody_loader_binds_canonical_marker_mode_hash_and_history(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    project = repo / "reconstruction_v2"
    freeze_path = project / acq.IMPLEMENTATION_FREEZE_RELATIVE_PATH
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_bytes(b"historical-freeze\n")
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.invalid"), cwd=repo, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=repo, check=True)
    subprocess.run(("git", "add", "reconstruction_v2"), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-qm", "pre-seed freeze"), cwd=repo, check=True)
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repo).decode().strip()
    freeze_hash = "a" * 64
    marker_body = {
        "schema": f"{acq.VERSION}_seed_generation_attempt_marker",
        "version": acq.VERSION,
        "status": "sole_seed_generation_attempt_consumed",
        "actual_HEAD": head,
        "implementation_freeze_sha256": freeze_hash,
        "design_sha256": acq.DESIGN_SHA256,
        "attempt_count": 1,
    }
    marker = {**marker_body, "marker_sha256": acq.semantic_hash(marker_body)}
    marker_path = project / acq.SEED_MARKER_RELATIVE_PATH
    marker_file_hash = acq._write_json_exclusive(marker_path, marker, acq.PUBLIC_MODE)
    custody_body = {
        "schema": acq.CUSTODY_SCHEMA,
        "version": acq.VERSION,
        "status": "seed_committed_cohort_not_generated",
        "design_sha256": acq.DESIGN_SHA256,
        "design_file_sha256": acq.DESIGN_FILE_SHA256,
        "grammar_sha256": acq.GRAMMAR_SHA256,
        "graph_core_sha256": acq.GRAPH_CORE_SHA256,
        "amendment_sha256": acq.AMENDMENT_SHA256,
        "implementation_freeze_sha256": freeze_hash,
        "seed_attempt_marker_sha256": marker["marker_sha256"],
        "seed_attempt_marker_file_sha256": marker_file_hash,
        "seed_bytes": acq.SEED_BYTES,
        "seed_generation": "os.urandom_exactly_once_after_marker_O_EXCL_mode_0600",
        "seed_commitment_sha256": "b" * 64,
        "seed_material_published": False,
        "cohort_generated": False,
        "seed_trials_allowed": 1,
    }
    custody = {**custody_body, "custody_sha256": acq.semantic_hash(custody_body)}
    custody_path = project / acq.SEED_CUSTODY_RELATIVE_PATH
    acq._write_json_exclusive(custody_path, custody, acq.PUBLIC_MODE)
    assert acq.load_seed_custody(custody_path) == custody
    marker_path.chmod(acq.PRIVATE_MODE)
    with pytest.raises(acq.SyntheticCausalAcquisitionError, match="mode or type"):
        acq.load_seed_custody(custody_path)


def test_acquisition_loader_binds_private_cohort_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    private_root = root / acq.PRIVATE_COHORT_RELATIVE_PATH
    private_root.mkdir(parents=True)
    for name in acq.EXPECTED_PRIVATE_COHORT_FILES:
        (private_root / name).write_bytes(b"sealed\n")
    freeze = {"implementation_freeze_sha256": "a" * 64}
    custody = {
        "custody_sha256": "b" * 64,
        "seed_commitment_sha256": "c" * 64,
    }
    custody_path = root / acq.SEED_CUSTODY_RELATIVE_PATH
    custody_path.parent.mkdir(parents=True, exist_ok=True)
    custody_path.write_bytes(acq.canonical_bytes(custody) + b"\n")
    marker_body = {
        "schema": acq.ATTEMPT_SCHEMA,
        "version": acq.VERSION,
        "status": "formal_cohort_attempt_consumed",
        "design_sha256": acq.DESIGN_SHA256,
        "custody_sha256": custody["custody_sha256"],
        "seed_commitment_sha256": custody["seed_commitment_sha256"],
        "actual_HEAD": "d" * 40,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "attempt_count": 1,
    }
    marker = {**marker_body, "marker_sha256": acq.semantic_hash(marker_body)}
    marker_path = root / acq.COHORT_MARKER_RELATIVE_PATH
    marker_file_hash = acq._write_json_exclusive(marker_path, marker, acq.PRIVATE_MODE)
    receipt_body = {
        "schema": acq.RECEIPT_SCHEMA,
        "status": "formal_cohort_acquired_private_labels_separated",
        "design_sha256": acq.DESIGN_SHA256,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "custody_sha256": custody["custody_sha256"],
        "seed_commitment_sha256": custody["seed_commitment_sha256"],
        "attempt_marker_sha256": marker["marker_sha256"],
        "attempt_marker_file_sha256": marker_file_hash,
        "F_search_labels_created": False,
        "total_count": 256,
        "packs": {block: {} for block in grammar.BLOCK_ORDER},
    }
    receipt = {**receipt_body, "receipt_sha256": acq.semantic_hash(receipt_body)}
    monkeypatch.setattr(acq, "verify_implementation_freeze", lambda _root: (freeze, "e" * 40))
    monkeypatch.setattr(
        acq,
        "_load_committed_public_json",
        lambda _root, relative, _field: (
            receipt if relative == acq.ACQUISITION_RECEIPT_RELATIVE_PATH else custody
        ),
    )
    monkeypatch.setattr(acq, "load_seed_custody", lambda _path: custody)
    monkeypatch.setattr(acq, "_require_ancestor", lambda _root, commit, _field: str(commit))
    monkeypatch.setattr(
        acq, "_historical_bytes", lambda _root, _commit, _relative: custody_path.read_bytes()
    )
    assert acq.load_committed_acquisition_receipt(root) == receipt
    marker_path.chmod(acq.PUBLIC_MODE)
    with pytest.raises(acq.SyntheticCausalAcquisitionError, match="mode or type"):
        acq.load_committed_acquisition_receipt(root)


def test_publication_terminal_loader_reuses_runner_recursive_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    receipt_path = root / runner.FORMATION_RECEIPT_RELATIVE_PATH
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(b"validated-formation\n")
    acquisition = {"receipt_sha256": "a" * 64}
    freeze = {"implementation_freeze_sha256": "b" * 64}
    formation = {
        "stage": "formation",
        "status": "terminal_unidentifiable_transition",
        "parent_receipt_sha256": None,
        "item_rows_or_item_commitments_persisted_publicly": False,
        "receipt_sha256": "c" * 64,
    }
    calls: list[str] = []
    monkeypatch.setattr(acq, "load_committed_acquisition_receipt", lambda _root: acquisition)
    monkeypatch.setattr(acq, "verify_implementation_freeze", lambda _root: (freeze, "d" * 40))

    def strict_loader(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["stage"]))
        assert kwargs["acquisition"] == acquisition
        assert kwargs["freeze"] == freeze
        return formation

    monkeypatch.setattr(runner, "_load_validated_stage_receipt", strict_loader)
    stage, observed, file_hash = acq._load_terminal_stage_receipt(root)
    assert (stage, observed, calls) == ("formation", formation, ["formation"])
    assert file_hash == acq.sha256_file(receipt_path)


def test_formal_entrypoints_are_canonical_and_marker_precedes_secret_access() -> None:
    import inspect

    assert tuple(inspect.signature(acq.create_seed_custody).parameters) == ("project_root",)
    assert tuple(inspect.signature(acq.acquire_formal_cohort).parameters) == ("project_root",)
    seed_source = inspect.getsource(acq.create_seed_custody)
    assert seed_source.index("_write_json_exclusive(seed_marker_path") < seed_source.index(
        "os.urandom"
    )
    cohort_source = inspect.getsource(acq.acquire_formal_cohort)
    assert cohort_source.index("_write_json_exclusive(") < cohort_source.index(
        "_read_private_seed"
    )
    publish_source = inspect.getsource(acq.publish_terminal_reproducibility)
    assert publish_source.index("_load_terminal_stage_receipt") < publish_source.index(
        "_read_private_seed"
    )
    cli_source = inspect.getsource(runner.main)
    assert "--minilm-manifest" not in cli_source
    assert "--attestation-receipt" not in cli_source
    assert "_prepare_formal_resources" in cli_source


def test_fake_resources_cannot_consume_canonical_stage() -> None:
    with pytest.raises(runner.SyntheticCausalRunnerError, match="formal CLI"):
        runner.run_canonical_stage(
            project_root=Path(__file__).resolve().parents[1],
            stage="formation",
            encoder=HashEncoder(),
            runtime=FakeOfficialRuntime(),
        )


def test_action_pack_projection_excludes_gold_and_bruteforceable_item_commitment(
    private_packs: dict[str, tuple[Path, Path | None]],
) -> None:
    view_path, _labels = private_packs["A_form"]
    payload = json.loads(view_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "gold_node_indices",
        "latent_role",
        "item_commitment_sha256",
        "label_free_commitment_sha256",
        "family_id",
        "family_role",
        "polarity",
        "pair_key",
        "matching_signature_sha256",
    ):
        assert forbidden not in serialized
    assert len(payload["rows"]) == 64
    assert len({row["opaque_view_sha256"] for row in payload["rows"]}) == 64


def test_post_terminal_reproducibility_projection_is_complete_but_has_no_outputs() -> None:
    item = grammar.generate_block(TEST_SEED, "A_form")[0]
    row = acq._compiled_public_row(item)
    assert row["gold_node_indices"] == list(item.gold_node_indices)
    assert len(row["nodes"]) == 32
    assert all("latent_role" in node for node in row["nodes"])
    serialized = json.dumps(row, sort_keys=True)
    assert "official_top5" not in serialized
    assert "Agent_full" not in serialized
    project_root = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        ("git", "check-ignore", acq.REPRODUCIBILITY_RELATIVE_PATH.as_posix()),
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert ignored.returncode == 1


def test_f_search_has_no_label_pack(
    private_packs: dict[str, tuple[Path, Path | None]],
) -> None:
    _view, labels = private_packs["F_search"]
    assert labels is None
    with pytest.raises(runner.SyntheticCausalRunnerError, match="forbidden"):
        runner.load_label_block(Path("never-open"), "F_search")


def test_loaders_reject_public_mode_and_tampered_full_graph(
    private_packs: dict[str, tuple[Path, Path | None]], tmp_path: Path
) -> None:
    view_path, _labels = private_packs["A_form"]
    public_copy = tmp_path / "public.json"
    public_copy.write_bytes(view_path.read_bytes())
    public_copy.chmod(0o644)
    with pytest.raises(runner.SyntheticCausalRunnerError, match="mode or size"):
        runner.load_view_block(public_copy, "A_form")

    payload = json.loads(view_path.read_text(encoding="utf-8"))
    payload["rows"][0]["edges_by_mode"][grammar.FULL_GRAPH] = []
    row = payload["rows"][0]
    body = dict(row)
    body.pop("opaque_view_sha256")
    row["opaque_view_sha256"] = acq.semantic_hash(body)
    block_body = dict(payload)
    block_body.pop("block_sha256")
    payload["block_sha256"] = acq.semantic_hash(block_body)
    tampered = tmp_path / "tampered.json"
    acq._write_json_exclusive(tampered, payload, acq.PRIVATE_MODE)
    with pytest.raises(runner.SyntheticCausalRunnerError, match="pinned core"):
        runner.load_view_block(tampered, "A_form")


def test_formation_is_label_late_f_label_free_and_uses_one_shared_official_wave(
    private_packs: dict[str, tuple[Path, Path | None]], tmp_path: Path
) -> None:
    a_view = runner.load_view_block(private_packs["A_form"][0], "A_form")
    f_view = runner.load_view_block(private_packs["F_search"][0], "F_search")
    label_path = private_packs["A_form"][1]
    assert label_path is not None
    runtime = FakeOfficialRuntime()
    opened_after_calls: list[int] = []
    seal_path = tmp_path / "formation.action.seal.json"

    def load_labels() -> runner.LabelBlock:
        opened_after_calls.append(runtime.calls)
        assert seal_path.is_file()
        return runner.load_label_block(label_path, "A_form")

    outcome = runner.run_formation(
        a_view,
        f_view,
        a_label_loader=load_labels,
        encoder=HashEncoder(),
        runtime=runtime,
        work_root=tmp_path / "formation-work",
        action_seal_path=seal_path,
    )
    assert runtime.calls == 128
    assert runtime.max_live <= runner.OFFICIAL_CONCURRENCY_CAP
    assert opened_after_calls == [128]
    assert outcome.real_evaluator_id in runner.EVALUATOR_IDS
    assert outcome.permuted_evaluator_id in runner.EVALUATOR_IDS
    assert outcome.real_recipe_id in runner.RECIPE_IDS
    receipt = runner.formation_public_receipt(outcome)
    assert receipt["F_search_labels_created_or_opened"] is False
    assert receipt["item_rows_or_item_commitments_persisted_publicly"] is False
    assert receipt["action_seal_file_sha256"] == acq.sha256_file(seal_path)
    assert receipt["evaluator_derangement_effective_same_gold_vector_count"] >= 0
    assert receipt["same_gold_vector_count_is_descriptive_not_a_gate_or_retry_trigger"] is True
    assert receipt["receipt_sha256"] == acq.semantic_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_measurement_reports_matched_action_evaluator_and_familyout_aggregates_only(
    private_packs: dict[str, tuple[Path, Path | None]], tmp_path: Path
) -> None:
    view = runner.load_view_block(private_packs["A_hold"][0], "A_hold")
    label_path = private_packs["A_hold"][1]
    assert label_path is not None
    runtime = FakeOfficialRuntime()
    opened_after_calls: list[int] = []
    seal_path = tmp_path / "A_hold.action.seal.json"

    def load_labels() -> runner.LabelBlock:
        opened_after_calls.append(runtime.calls)
        assert seal_path.is_file()
        return runner.load_label_block(label_path, "A_hold")

    outcome = runner.run_measurement(
        view,
        real_recipe_id="R8_ALL_TYPED_2SWAP",
        permuted_recipe_id="R1_DEFINITION_1SWAP",
        fixed_e00_recipe_id="R2_EXCEPTION_1SWAP",
        label_loader=load_labels,
        encoder=HashEncoder(),
        runtime=runtime,
        work_root=tmp_path / "measurement-work",
        action_seal_path=seal_path,
    )
    assert runtime.calls == 64
    assert opened_after_calls == [64]
    assert set(outcome.mechanism_reference_tests) == {
        "full_minus_drop_designated_positive_minus_negative",
        "full_minus_wrong_type_positive_minus_negative",
        "full_minus_endpoint_permuted_positive_minus_negative",
    }
    assert outcome.primary_reference_test["design_based_randomization_p_value"] is False
    assert "protocol_promoted" in outcome.primary_reference_test
    assert all(
        "protocol_promoted" not in value
        for value in (*outcome.mechanism_reference_tests.values(), *outcome.evaluator_reference_tests.values())
    )
    assert set(outcome.aggregates["Agent_full"]["by_edge_family"]) == set(
        grammar.EDGE_FAMILIES
    )
    assert Counter(
        summary["item_count"]
        for summary in outcome.aggregates["Agent_full"]["by_family_id"].values()
    ) == {4: 16}
    receipt = runner.measurement_public_receipt(outcome)
    serialized = json.dumps(receipt, sort_keys=True)
    assert "gold_node_indices" not in serialized
    assert "item_commitment_sha256" not in serialized
    assert receipt["reference_tail_is_design_based_randomization_p_value"] is False


def test_m_search_is_sealed_without_promotion(
    private_packs: dict[str, tuple[Path, Path | None]], tmp_path: Path
) -> None:
    with pytest.raises(runner.SyntheticCausalRunnerError, match="remains sealed"):
        runner.run_m_if_authorized(
            authorized=False,
            view_loader=lambda: runner.load_view_block(
                private_packs["M_search"][0], "M_search"
            ),
            label_loader=lambda: runner.load_label_block(
                private_packs["M_search"][1], "M_search"  # type: ignore[arg-type]
            ),
            real_recipe_id="R8_ALL_TYPED_2SWAP",
            permuted_recipe_id="R1_DEFINITION_1SWAP",
            fixed_e00_recipe_id="R2_EXCEPTION_1SWAP",
            encoder=HashEncoder(),
            runtime=FakeOfficialRuntime(),
            work_root=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()
