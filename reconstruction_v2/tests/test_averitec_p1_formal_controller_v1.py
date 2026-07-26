from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

from assumption_agent.benchmarks import averitec_p1_acquisition_v1 as acquisition
from assumption_agent.benchmarks import averitec_p1_coordinate_worker_v1 as coordinate
from assumption_agent.benchmarks import averitec_p1_formal_controller_v1 as controller
from assumption_agent.benchmarks import averitec_p1_typed_core_v1 as core


def _write_private(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(acquisition.canonical_bytes(value))
    path.chmod(0o600)


def _family_rows(block: str, item_ids: list[str]) -> list[dict[str, object]]:
    per_family = controller.FORMAL_FAMILY_COUNTS[block]
    families = [
        family
        for family in acquisition.FAMILIES
        for _ in range(per_family[family])
    ]
    return [
        {
            "family": family,
            "item_id": item_id,
            "qrel_document_ordinals": [1, 5],
        }
        for item_id, family in zip(item_ids, families)
    ]


def _make_acquisition(root: Path) -> None:
    corpus = [
        {
            "body": f"synthetic body {ordinal}",
            "document_id": hashlib.sha256(
                f"document {ordinal}".encode()
            ).hexdigest(),
            "ordinal": ordinal,
            "title": f"synthetic title {ordinal}",
        }
        for ordinal in range(6)
    ]
    for block in acquisition.BLOCK_ORDER:
        count = controller.FORMAL_BLOCK_QUERY_COUNTS[block]
        queries = [
            {
                "item_id": hashlib.sha256(
                    f"{block} item {ordinal}".encode()
                ).hexdigest(),
                "ordinal": ordinal,
                "text": f"synthetic {block} claim {ordinal}",
            }
            for ordinal in range(count)
        ]
        view = acquisition.self_hashed(
            {
                "block": block,
                "corpus": corpus,
                "queries": queries,
                "schema": (
                    f"{acquisition.VERSION}_label_free_action_view_v1"
                ),
                "study_id": core.STUDY_ID,
            }
        )
        _write_private(root / f"{block}.view.json", view)
        if block != acquisition.F_SEARCH:
            qrels = acquisition.self_hashed(
                {
                    "block": block,
                    "rows": _family_rows(
                        block, [str(row["item_id"]) for row in queries]
                    ),
                    "schema": f"{acquisition.VERSION}_late_qrel_pack_v1",
                    "study_id": core.STUDY_ID,
                }
            )
            _write_private(root / f"{block}.qrels.json", qrels)


class _CoordinateExecutor:
    def __init__(self) -> None:
        self.blocks: list[str] = []

    def __call__(self, *, block, private_input):
        self.blocks.append(block)
        documents, queries = coordinate.validate_input(private_input)
        direct = [900_000, 800_000, 700_000, 600_000, 500_000, 400_000]
        scores = {
            core.DIRECT: direct,
            core.CAUSE: [100_000, 100_000, 100_000, 100_000, 100_000, 990_000],
            core.EFFECT: [100_000, 980_000, 100_000, 100_000, 100_000, 100_000],
            core.QUOTE: [100_000, 100_000, 100_000, 100_000, 970_000, 100_000],
            core.SOURCE: [100_000, 100_000, 960_000, 100_000, 100_000, 100_000],
            core.NUMBER: [100_000, 100_000, 100_000, 950_000, 100_000, 100_000],
            core.COMPARE: [940_000, 100_000, 100_000, 100_000, 100_000, 100_000],
            core.CONTEXT: [100_000, 100_000, 100_000, 930_000, 100_000, 100_000],
        }
        assert len(documents) == 6
        body = {
            "document_count": len(documents),
            "input_sha256": coordinate.stable_hash(private_input),
            "query_count": len(queries),
            "rows": [
                {
                    "item_id": item_id,
                    "variant_scores": {
                        variant: list(scores[variant])
                        for variant in core.QUERY_VARIANT_IDS
                    },
                }
                for item_id, _text in queries
            ],
            "runtime_receipt": {
                "cuda_allocate_and_synchronize": True,
                "cuda_device_count": 1,
                "cuda_logical_device": 0,
                "deterministic_algorithms_enabled": True,
                "minilm_all_parameters_cuda0": True,
                "minilm_parameter_count": 1,
                "native_and_torch_thread_count": 1,
                "torch_manual_seed": 0,
            },
            "schema": coordinate.OUTPUT_SCHEMA,
            "study_id": core.STUDY_ID,
        }
        body["self_sha256"] = coordinate.stable_hash(body)
        return body


class _HippoExecutor:
    def __init__(self) -> None:
        self.blocks: list[str] = []

    def __call__(self, *, block, articles, queries):
        self.blocks.append(block)
        assert [article["idx"] for article in articles] == list(range(6))
        return controller.HippoResult(
            indices=tuple((0, 1, 2, 3, 4) for _item, _text in queries),
            receipt_sha256="a" * 64,
            build_receipt_sha256="b" * 64,
        )


class _NoGainCoordinateExecutor(_CoordinateExecutor):
    def __call__(self, *, block, private_input):
        output = super().__call__(block=block, private_input=private_input)
        for row in output["rows"]:
            direct = list(row["variant_scores"][core.DIRECT])
            row["variant_scores"] = {
                variant: list(direct) for variant in core.QUERY_VARIANT_IDS
            }
        output.pop("self_sha256")
        output["self_sha256"] = coordinate.stable_hash(output)
        return output


def test_formal_controller_seals_then_promotes_and_opens_m_search(
    tmp_path: Path,
) -> None:
    del tmp_path  # Windows-mounted pytest temp roots do not preserve POSIX mode.
    with tempfile.TemporaryDirectory(
        prefix="averitec-controller-", dir="/tmp"
    ) as temporary:
        root = Path(temporary)
        acquisition_root = root / "acquisition"
        acquisition_root.mkdir()
        _make_acquisition(acquisition_root)
        coordinates = _CoordinateExecutor()
        hippo = _HippoExecutor()
        formal = controller.FormalController(
            acquisition_root=acquisition_root,
            work_root=root / "work",
            execution_binding_sha256="c" * 64,
            coordinate_executor=coordinates,
            hippo_executor=hippo,
        )
        terminal = formal.run()
        assert terminal["status"] == "formal_lifecycle_complete"
        assert terminal["A_hold_evaluator_promoted"] is True
        assert (
            terminal["A_hold_reality_three_family_double_baseline_passed"]
            is True
        )
        assert terminal["M_search_L5_passed"] is True
        assert terminal["F_search_decision_or_gate_count"] == 0
        assert terminal["F_search_qrel_open_count"] == 0
        assert coordinates.blocks == list(acquisition.BLOCK_ORDER)
        assert hippo.blocks == [acquisition.A_HOLD]
        assert (
            root / "work" / "stages" / acquisition.M_SEARCH
            / "actions.private.json"
        ).is_file()
        assert not (acquisition_root / "F_search.qrels.json").exists()


def test_nonpromotion_never_opens_or_materializes_m_search(
    tmp_path: Path,
) -> None:
    del tmp_path
    with tempfile.TemporaryDirectory(
        prefix="averitec-no-promotion-", dir="/tmp"
    ) as temporary:
        root = Path(temporary)
        acquisition_root = root / "acquisition"
        acquisition_root.mkdir()
        _make_acquisition(acquisition_root)
        (acquisition_root / "M_search.view.json").unlink()
        (acquisition_root / "M_search.qrels.json").unlink()
        coordinates = _NoGainCoordinateExecutor()
        hippo = _HippoExecutor()
        formal = controller.FormalController(
            acquisition_root=acquisition_root,
            work_root=root / "work",
            execution_binding_sha256="d" * 64,
            coordinate_executor=coordinates,
            hippo_executor=hippo,
        )
        terminal = formal.run()
        assert terminal["status"] == "terminal_A_hold_E1_not_promoted"
        assert terminal["A_hold_evaluator_promoted"] is False
        assert terminal["M_search_L5_passed"] is None
        assert coordinates.blocks == [
            acquisition.A_FORM,
            acquisition.F_SEARCH,
            acquisition.A_HOLD,
        ]
        assert not (
            root / "work" / "stages" / acquisition.M_SEARCH
        ).exists()
