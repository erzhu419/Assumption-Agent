from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from assumption_agent.benchmarks import (
    gscl_arn_formal_item_factory_v1 as factory_module,
)
from assumption_agent.benchmarks.gscl_arn_formal_item_factory_v1 import (
    PRIVATE_OUTPUT_SCHEMA,
    FormalItemFactoryError,
    FrozenArnItemFactory,
    PrivateFactoryItemOutput,
    build_private_four_arm_output_qualification_only,
    require_internal_formal_output,
    run_formal_factory_files,
)
from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from assumption_agent.gscl_arn_intrinsic_scorers_v1 import (
    FrozenNarrativeScorers,
    IntrinsicScorerError,
)
from replication_runtime.qasper_minilm_v1.binding import EMBEDDING_DIMENSION
from replication_runtime.gscl_narrative_extractor_v1 import contract as extractor_contract
from replication_runtime.gscl_minilm_portable_v1.binding import (
    GSCLPortableOfflineMiniLMEncoder,
)


def _completion(left: str, verb: str, right: str) -> str:
    return json.dumps(
        {
            "generators": [
                {
                    "anchor_mention_id": "a0",
                    "causal_orientation": "forward",
                    "generator_id": "g0",
                    "generator_kind": "causal",
                    "polarity": "positive",
                    "slot_mention_ids": ["m0", "m1"],
                    "temporal_orientation": "forward",
                }
            ],
            "mentions": [
                {
                    "kind": "object",
                    "mention_id": "m0",
                    "occurrence": 0,
                    "quote": left,
                },
                {
                    "kind": "object",
                    "mention_id": "m1",
                    "occurrence": 0,
                    "quote": right,
                },
                {
                    "kind": "generator",
                    "mention_id": "a0",
                    "occurrence": 0,
                    "quote": verb,
                },
            ],
            "schema_version": "gscl.narrative.extraction.v1",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _extraction(source_id: str, left: str, verb: str, right: str):
    story = f"{left} {verb} {right}."
    return parse_untrusted_generator_completion(
        NarrativeSource(source_id, story),
        _completion(left, verb, right),
    )


class _Tokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text) + 2))}


class _Model:
    tokenizer = _Tokenizer()


class _Encoder:
    _model = _Model()
    runtime_receipt = {"runtime": "synthetic"}
    canary_receipt = {"canary": "synthetic"}

    def encode(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.asarray(
                [
                    ((digest[index % len(digest)] + index) % 251) + 1
                    for index in range(EMBEDDING_DIMENSION)
                ],
                dtype=np.float32,
            )
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows).astype(np.float32)


def _factory() -> tuple[FrozenArnItemFactory, object, tuple[object, object]]:
    query = _extraction("private.query", "Aster", "guides", "Birch")
    candidates = (
        _extraction(
            "private.first", "Cedar", "guides", "Dune"
        ),
        _extraction(
            "private.second", "Ember", "opposes", "Fjord"
        ),
    )
    factory = FrozenArnItemFactory._source_free_qualification(
        extractions=(query, *candidates), encoder=_Encoder()
    )
    return factory, query, candidates


def test_factory_derives_commitments_and_recomputes_all_four_arms() -> None:
    factory, query, candidates = _factory()
    output = factory.evaluate_private_item(
        opaque_item_id=hashlib.sha256(b"private item").hexdigest(),
        query=query,
        candidates=candidates,
    )
    assert output.recomputation_receipt["deep_recomputation_exact"] is True
    assert {row["arm_id"] for row in output.prediction_rows} == {
        "semantic_only",
        "legacy_keyword",
        "flat_label_no_verifier",
        "full_gscl",
    }
    assert factory.factory_receipt[
        "caller_supplied_commitments_accepted"
    ] is False
    assert factory.factory_receipt[
        "caller_supplied_predictions_accepted"
    ] is False
    assert factory.factory_receipt[
        "same_process_objects_are_security_boundary"
    ] is False


def test_qualification_and_forged_outputs_cannot_become_formal() -> None:
    factory, query, candidates = _factory()
    output = factory.evaluate_private_item(
        opaque_item_id=hashlib.sha256(b"private item").hexdigest(),
        query=query,
        candidates=candidates,
    )
    with pytest.raises(
        FormalItemFactoryError,
        match="external_or_qualification_output_rejected",
    ):
        require_internal_formal_output(output)
    forged = PrivateFactoryItemOutput(
        opaque_item_id=output.opaque_item_id,
        prediction_rows=output.prediction_rows,
        recomputation_receipt=output.recomputation_receipt,
        lineage="formal_frozen_assets",
        _token=object(),
    )
    with pytest.raises(
        FormalItemFactoryError,
        match="external_or_qualification_output_rejected",
    ):
        require_internal_formal_output(forged)


def test_item_factory_api_has_no_external_result_or_commitment_inputs() -> None:
    parameters = set(
        inspect.signature(
            FrozenArnItemFactory.evaluate_private_item
        ).parameters
    )
    assert parameters == {
        "self",
        "opaque_item_id",
        "query",
        "candidates",
    }
    forbidden = {
        "prepared",
        "predictions",
        "commitments",
        "mapping_result",
        "score_table",
    }
    assert not parameters.intersection(forbidden)


def test_formal_file_bridge_requires_qualified_target_manifest() -> None:
    parameters = set(
        inspect.signature(run_formal_factory_files).parameters
    )
    assert parameters == {
        "predictor_path",
        "batch_manifest_path",
        "minilm_manifest_path",
        "minilm_model_root",
        "minilm_target_manifest_path",
        "output_path",
    }
    source = inspect.getsource(run_formal_factory_files)
    assert "minilm_target_manifest_path" in source
    assert '"item_indices"' in source
    build_source = inspect.getsource(
        __import__(
            (
                "assumption_agent.benchmarks."
                "gscl_arn_formal_item_factory_v1"
            ),
            fromlist=["_build_private_four_arm_output"],
        )._build_private_four_arm_output
    )
    assert "FrozenArnItemFactory.from_frozen_assets" in build_source
    assert "OfflineMiniLMEncoder(" not in build_source


def test_private_output_writer_uses_exclusive_owned_file() -> None:
    root = Path(
        tempfile.mkdtemp(prefix="gscl-factory-output-", dir="/var/tmp")
    )
    root.chmod(0o700)
    try:
        output = root / "factory.json"
        factory_module._write_private_output_once(  # noqa: SLF001
            output, {"schema": "synthetic"}
        )
        assert output.read_bytes() == b'{"schema":"synthetic"}\n'
        assert output.stat().st_mode & 0o777 == 0o600
        with pytest.raises(
            FormalItemFactoryError,
            match="factory_output_already_exists",
        ):
            factory_module._write_private_output_once(  # noqa: SLF001
                output, {"schema": "changed"}
            )
    finally:
        shutil.rmtree(root)


def test_formal_factory_constructs_exact_target_local_portable_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_init(
        self: GSCLPortableOfflineMiniLMEncoder,
        **kwargs: object,
    ) -> None:
        calls.append(dict(kwargs))
        self._encoder = _Encoder()
        self.runtime_receipt = {
            "schema": "gscl_minilm_portable_runtime_receipt_v1",
            "target_manifest_self_sha256": hashlib.sha256(
                b"target manifest"
            ).hexdigest(),
        }
        self.canary_receipt = {
            "schema": "gscl_minilm_portable_canary_receipt_v1",
            "repeat_count": 2,
            "repeat_byte_exact": True,
        }

    monkeypatch.setattr(
        GSCLPortableOfflineMiniLMEncoder, "__init__", fake_init
    )
    monkeypatch.setattr(
        GSCLPortableOfflineMiniLMEncoder,
        "validate_internal",
        lambda self: None,
    )
    query = _extraction(
        "private.formal.query", "Aster", "guides", "Birch"
    )
    candidates = (
        _extraction(
            "private.formal.first", "Cedar", "guides", "Dune"
        ),
        _extraction(
            "private.formal.second", "Ember", "opposes", "Fjord"
        ),
    )
    factory = FrozenArnItemFactory.from_frozen_assets(
        extractions=(query, *candidates),
        asset_manifest_path=Path("/public/asset.json"),
        model_root=Path("/public/model"),
        target_manifest_path=Path("/private/target.json"),
    )
    assert calls == [
        {
            "asset_manifest_path": Path("/public/asset.json"),
            "model_root": Path("/public/model"),
            "target_manifest_path": Path("/private/target.json"),
            "run_canary": True,
        }
    ]
    assert factory.lineage == "formal_frozen_assets"
    assert factory.scorers.receipt["construction_domain"] == (
        "formal_exact_gscl_target_local_portable_minilm_v1"
    )
    assert factory.factory_receipt["encoder_exact_type"].endswith(
        ".GSCLPortableOfflineMiniLMEncoder"
    )
    assert json.loads(
        factory.factory_receipt["encoder_runtime_receipt_json"]
    )["schema"] == "gscl_minilm_portable_runtime_receipt_v1"
    assert json.loads(
        factory.factory_receipt["encoder_canary_receipt_json"]
    )["repeat_byte_exact"] is True


def test_factory_rejects_mutated_internal_scorer_state() -> None:
    factory, query, candidates = _factory()
    vector = next(iter(factory.scorers.source_vectors.values()))
    with pytest.raises(ValueError):
        vector[0] = vector[0] + np.float32(0.125)
    output = factory.evaluate_private_item(
        opaque_item_id=hashlib.sha256(b"private item").hexdigest(),
        query=query,
        candidates=candidates,
    )
    assert output.recomputation_receipt["deep_recomputation_exact"] is True


def test_source_free_extractor_batches_feed_internal_four_arm_factory() -> None:
    opaque = hashlib.sha256(b"opaque synthetic item").hexdigest()
    stories = (
        ("Aster", "guides", "Birch"),
        ("Cedar", "guides", "Dune"),
        ("Ember", "opposes", "Fjord"),
    )
    predictor = {
        "rows": [
            {
                "opaque_item_id": opaque,
                "query_narrative": "Aster guides Birch.",
                "first_choice": "Cedar guides Dune.",
                "second_choice": "Ember opposes Fjord.",
            }
        ]
    }
    predictor_raw = (
        json.dumps(
            predictor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    input_raw = extractor_contract.encode_input(
        batch_id="synthetic-batch",
        sequence=0,
        requests=tuple(
            extractor_contract.StoryRequest(
                ordinal=index,
                story_text=f"{left} {verb} {right}.",
            )
            for index, (left, verb, right) in enumerate(stories)
        ),
    )
    pack = extractor_contract.admit_story_only_pack_qualification_only(
        input_raw
    )
    results = [
        extractor_contract.valid_result(
            ordinal=index,
            story_commitment=pack.story_commitments[index],
            completion=_completion(left, verb, right),
            completion_token_count=32,
        )
        for index, (left, verb, right) in enumerate(stories)
    ]
    closure = extractor_contract.ExecutionClosure(
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        parser_closure_sha256=hashlib.sha256(b"parser").hexdigest(),
        model_asset_manifest_sha256=hashlib.sha256(
            b"model asset"
        ).hexdigest(),
        model_runtime_closure_sha256=hashlib.sha256(
            b"model runtime"
        ).hexdigest(),
        target_double_run_receipt_sha256=hashlib.sha256(
            b"double run"
        ).hexdigest(),
    )
    output_raw = extractor_contract.encode_private_output(
        pack=pack,
        execution_closure=closure,
        results=results,
    )
    result = build_private_four_arm_output_qualification_only(
        predictor_raw=predictor_raw,
        input_batch_raws=(input_raw,),
        output_batch_raws=(output_raw,),
        encoder=_Encoder(),
    )
    assert result["schema"] == PRIVATE_OUTPUT_SCHEMA
    assert result["lineage"] == "synthetic_source_free_qualification"
    assert set(result["by_arm"]) == {
        "semantic_only",
        "legacy_keyword",
        "flat_label_no_verifier",
        "full_gscl",
    }
    assert all(len(rows) == 1 for rows in result["by_arm"].values())
    assert result["caller_predictions_accepted"] is False


def test_formal_batch_item_indices_bind_round_robin_triplets() -> None:
    rows = [
        {
            "opaque_item_id": f"{index}" * 64,
            "query_narrative": f"Query {index} alpha beta gamma.",
            "first_choice": f"First {index} delta epsilon zeta.",
            "second_choice": f"Second {index} eta theta iota.",
        }
        for index in range(4)
    ]
    expected = factory_module._expected_stories(rows)  # noqa: SLF001
    triplets = tuple(
        tuple(expected[offset : offset + 3])
        for offset in range(0, len(expected), 3)
    )
    closure = extractor_contract.ExecutionClosure(
        prompt_sha256="1" * 64,
        parser_closure_sha256="2" * 64,
        model_asset_manifest_sha256="3" * 64,
        model_runtime_closure_sha256="4" * 64,
        target_double_run_receipt_sha256="5" * 64,
    )
    batches: list[tuple[object, ...]] = []
    for sequence, item_indices in enumerate(((0, 2), (1, 3))):
        stories = tuple(
            story
            for item_index in item_indices
            for story in triplets[item_index]
        )
        input_raw = extractor_contract.encode_input(
            batch_id=f"round-robin-{sequence}",
            sequence=sequence,
            requests=tuple(
                extractor_contract.StoryRequest(
                    ordinal=ordinal,
                    story_text=story,
                )
                for ordinal, (_, _, story) in enumerate(stories)
            ),
        )
        pack = (
            extractor_contract.admit_story_only_pack_qualification_only(
                input_raw
            )
        )
        output_raw = extractor_contract.encode_private_output(
            pack=pack,
            execution_closure=closure,
            results=tuple(
                extractor_contract.invalid_result(
                    ordinal=ordinal,
                    story_commitment=pack.story_commitments[ordinal],
                    error_code="COMPLETION_INVALID",
                )
                for ordinal in range(len(stories))
            ),
        )
        batches.append(
            (
                pack,
                extractor_contract.decode_private_output(
                    output_raw, expected_pack=pack
                ),
                hashlib.sha256(output_raw).hexdigest(),
                item_indices,
            )
        )
    extracted, invalid, receipts = (
        factory_module._decode_extractor_batches(  # noqa: SLF001
            predictor_rows=rows,
            batches=tuple(batches),
        )
    )
    assert extracted == {}
    assert invalid == {row["opaque_item_id"] for row in rows}
    assert [row["story_count"] for row in receipts] == [6, 6]
    with pytest.raises(
        FormalItemFactoryError,
        match="extractor_batch_item_indices_invalid",
    ):
        factory_module._decode_extractor_batches(  # noqa: SLF001
            predictor_rows=rows,
            batches=(batches[0], (*batches[1][:3], (1, 2))),
        )


def test_formal_all_invalid_still_attests_exact_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_init(
        self: GSCLPortableOfflineMiniLMEncoder,
        **kwargs: object,
    ) -> None:
        calls.append(dict(kwargs))
        self.runtime_receipt = {"schema": "synthetic-runtime"}
        self.canary_receipt = {"schema": "synthetic-canary"}

    monkeypatch.setattr(
        GSCLPortableOfflineMiniLMEncoder, "__init__", fake_init
    )
    opaque = hashlib.sha256(b"all-invalid-item").hexdigest()
    story_texts = (
        "Aster guides Birch.",
        "Cedar guides Dune.",
        "Ember opposes Fjord.",
    )
    predictor_raw = (
        json.dumps(
            {
                "rows": [
                    {
                        "opaque_item_id": opaque,
                        "query_narrative": story_texts[0],
                        "first_choice": story_texts[1],
                        "second_choice": story_texts[2],
                    }
                ]
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    input_raw = extractor_contract.encode_input(
        batch_id="all-invalid-batch",
        sequence=0,
        requests=tuple(
            extractor_contract.StoryRequest(
                ordinal=index, story_text=story
            )
            for index, story in enumerate(story_texts)
        ),
    )
    pack = extractor_contract.admit_story_only_pack_qualification_only(
        input_raw
    )
    output_raw = extractor_contract.encode_private_output(
        pack=pack,
        execution_closure=extractor_contract.ExecutionClosure(
            prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
            parser_closure_sha256=hashlib.sha256(b"parser").hexdigest(),
            model_asset_manifest_sha256=hashlib.sha256(
                b"model"
            ).hexdigest(),
            model_runtime_closure_sha256=hashlib.sha256(
                b"runtime"
            ).hexdigest(),
            target_double_run_receipt_sha256=hashlib.sha256(
                b"double"
            ).hexdigest(),
        ),
        results=tuple(
            extractor_contract.invalid_result(
                ordinal=index,
                story_commitment=pack.story_commitments[index],
                error_code="COMPLETION_INVALID",
            )
            for index in range(len(story_texts))
        ),
    )
    decoded = extractor_contract.decode_private_output(
        output_raw, expected_pack=pack
    )
    result = factory_module._build_private_four_arm_output(  # noqa: SLF001
        predictor_raw=predictor_raw,
        batches=(
            (
                pack,
                decoded,
                hashlib.sha256(output_raw).hexdigest(),
            ),
        ),
        asset_manifest_path=Path("/public/asset.json"),
        model_root=Path("/public/model"),
        target_manifest_path=Path("/private/target.json"),
        qualification_encoder=None,
    )
    assert calls == [
        {
            "asset_manifest_path": Path("/public/asset.json"),
            "model_root": Path("/public/model"),
            "target_manifest_path": Path("/private/target.json"),
            "run_canary": True,
        }
    ]
    assert result["error_item_count"] == result["item_count"] == 1
    assert result["factory_receipt_self_hash"] is None
    assert result["encoder_binding"]["encoder_exact_type"].endswith(
        ".GSCLPortableOfflineMiniLMEncoder"
    )
    assert all(
        rows == [
            {
                "disposition": "ERROR",
                "error_code": "ARM_RUNTIME_ERROR",
                "opaque_item_id": opaque,
                "selected_choice": None,
            }
        ]
        for rows in result["by_arm"].values()
    )


def test_scorer_subclass_cannot_cross_internal_factory_check() -> None:
    factory, query, candidates = _factory()

    class EvilScorers(FrozenNarrativeScorers):
        pass

    with pytest.raises(
        IntrinsicScorerError, match="scorer_subclass_forbidden"
    ):
        EvilScorers(
            source_vectors=factory.scorers.source_vectors,
            mention_vectors=factory.scorers.mention_vectors,
            primed_extraction_hashes=(
                factory.scorers.primed_extraction_hashes
            ),
            receipt=factory.scorers.receipt,
            _construction_token=object(),
        )
