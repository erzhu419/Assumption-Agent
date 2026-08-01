from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pytest

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_worker as closed_v1,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    contract as contract_v1,
)
from replication_runtime.gscl_narrative_extractor_v1 import worker
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as closed_v2,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    fixed_public_integrated_qualification as integrated,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    fixed_public_qualification as extractor,
)


_RUNTIME_COMMITMENT = hashlib.sha256(
    b"fixed-integrated-qualification-runtime"
).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    if isinstance(value, Mapping):
        value = dict(value)
    return contract_v1.canonical_json_bytes(value)


def _manifest(root: Path) -> worker.ModelAssetManifest:
    model_root = root / "model"
    model_root.mkdir(mode=0o700, parents=True)
    payload = b"fixed integrated synthetic model\n"
    path = model_root / "model.safetensors"
    path.write_bytes(payload)
    path.chmod(0o600)
    files = (
        {
            "path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        },
    )
    return worker.ModelAssetManifest(
        declarations={},
        files=files,
        runtime_requirements={},
        tree_sha256=contract_v1.semantic_sha256(list(files)),
        self_sha256=hashlib.sha256(b"manifest-self").hexdigest(),
        manifest_file_sha256=hashlib.sha256(
            b"manifest-file"
        ).hexdigest(),
        _marker=worker._VERIFIED_MANIFEST_MARKER,
    )


def _manifest_commitments(
    manifest: worker.ModelAssetManifest,
) -> dict[str, str]:
    return {
        "manifest_file_sha256": (
            manifest.manifest_file_sha256
        ),
        "manifest_self_sha256": manifest.self_sha256,
        "model_tree_sha256": manifest.tree_sha256,
    }


def _upstream_receipt(
    path: Path, manifest: worker.ModelAssetManifest
) -> Path:
    implementation = extractor._implementation_closure()
    body: dict[str, object] = {
        "counters": integrated._zero_counters(),
        "fixture_commitments": dict(
            extractor.FIXTURE_COMMITMENTS
        ),
        "fixture_count": len(extractor.PUBLIC_FIXTURES),
        "fixture_ordinals": list(
            range(len(extractor.PUBLIC_FIXTURES))
        ),
        "fixture_suite_sha256": extractor.FIXTURE_SUITE_SHA256,
        "implementation_closure": implementation,
        "implementation_closure_sha256": (
            contract_v1.semantic_sha256(implementation)
        ),
        "manifest_commitments": _manifest_commitments(manifest),
        "outcome_counts": {
            "success": len(extractor.PUBLIC_FIXTURES),
            "typed_abstention": 0,
            "typed_error": 0,
        },
        "qualification_passed": True,
        "repeat_byte_exact": True,
        "repeat_count": extractor.REPEAT_COUNT,
        "runtime_commitment": _RUNTIME_COMMITMENT,
        "schema": extractor.AGGREGATE_RECEIPT_SCHEMA,
    }
    receipt = {
        **body,
        "self_sha256": contract_v1.semantic_sha256(body),
    }
    path.write_bytes(_canonical_bytes(receipt))
    path.chmod(0o600)
    return path


class _Backend:
    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def score_batch(
        self, pairs: tuple[closed_v1.PromptAnswer, ...]
    ) -> tuple[closed_v2.TeacherForcedScore, ...]:
        rows: list[closed_v2.TeacherForcedScore] = []
        for pair in pairs:
            preferred = int(
                pair.candidate_key.endswith(
                    ".plan.one_relation"
                )
            )
            answer_tokens = max(1, len(pair.answer.split()))
            rows.append(
                closed_v2.TeacherForcedScore(
                    total_logprob_microunits=(
                        preferred * 1_000_000 * answer_tokens
                    ),
                    answer_token_count=answer_tokens,
                    context_and_answer_token_count=(
                        answer_tokens + 64
                    ),
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        return max(1, len(completion.encode("utf-8")) // 4)


def _parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource(
            "fixed.integrated."
            + hashlib.sha256(
                story.encode("utf-8")
            ).hexdigest()[:24],
            story,
        ),
        completion,
    )


class _Runtime:
    def __init__(self, *, drift: bool = False) -> None:
        self.calls: Counter[str] = Counter()
        self.drift = drift

    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def select_story(
        self, story_text: str
    ) -> closed_v2.ClosedChoiceV2Decision:
        digest = hashlib.sha256(
            story_text.encode("utf-8")
        ).hexdigest()
        self.calls[digest] += 1
        decision = closed_v2.select_hierarchical_qualification_only(
            story_text,
            backend=_Backend(),
            narrative_parser=_parser,
        )
        if self.drift and self.calls[digest] % 2 == 0:
            body = {
                key: child
                for key, child in decision.receipt.items()
                if key != "self_sha256"
            }
            body["synthetic_repeat_drift"] = True
            receipt = {
                **body,
                "self_sha256": contract_v1.semantic_sha256(
                    body
                ),
            }
            return closed_v2.ClosedChoiceV2Decision(
                wire_completion=decision.wire_completion,
                canonical_completion=(
                    decision.canonical_completion
                ),
                extraction=decision.extraction,
                selected_answer_token_count=(
                    decision.selected_answer_token_count
                ),
                wire_completion_token_count=(
                    decision.wire_completion_token_count
                ),
                receipt_bytes=_canonical_bytes(receipt),
            )
        return decision


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runtime: _Runtime | None = None,
) -> tuple[Mapping[str, object], _Runtime]:
    manifest = _manifest(tmp_path)
    model_root = tmp_path / "model"
    upstream = _upstream_receipt(
        tmp_path / "upstream.safe.json", manifest
    )
    selected = runtime or _Runtime()
    monkeypatch.setattr(
        extractor,
        "_verify_model_binding",
        lambda *, model_root, manifest: None,
    )
    monkeypatch.setattr(
        extractor,
        "_load_exact_runtime",
        lambda *, model_root, manifest: selected,
    )
    output = tmp_path / "output"
    receipt = (
        integrated.run_fixed_public_integrated_qualification(
            model_root=model_root,
            manifest=manifest,
            upstream_aggregate_receipt=upstream,
            output_root=output,
        )
    )
    return receipt, selected


def _nested_strings(value: object) -> tuple[str, ...]:
    rows: list[str] = []
    if isinstance(value, str):
        rows.append(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            rows.extend(_nested_strings(key))
            rows.extend(_nested_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            rows.extend(_nested_strings(child))
    return tuple(rows)


def test_surface_is_fixed_and_has_no_story_scorer_or_label_input() -> None:
    signature = inspect.signature(
        integrated.run_fixed_public_integrated_qualification
    )
    assert tuple(signature.parameters) == (
        "model_root",
        "manifest",
        "upstream_aggregate_receipt",
        "output_root",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not {
        "answer",
        "candidate",
        "label",
        "query",
        "scorer",
        "source",
        "story",
        "text",
    } & set(signature.parameters)


def test_integrated_happy_path_is_repeat_exact_and_shared(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt, runtime = _run(monkeypatch, tmp_path)
    assert receipt["qualification_passed"] is True
    assert receipt["extractor_repeat_byte_exact"] is True
    assert receipt["arm_repeat_byte_exact"] is True
    assert receipt["arm_contract_passed"] is True
    assert receipt["full_checker_candidate_count"] == 2
    assert receipt["arm_core_version"] == (
        "gscl.arn.intrinsic.arms.v2.unit_mapping"
    )
    assert receipt["arm_name_mapping"] == {
        "flat": "flat_label_no_verifier",
        "full": "full_gscl",
        "legacy": "legacy_keyword",
        "semantic_only": "semantic_only",
    }
    assert receipt["fixture_ordinals"] == [0, 1, 2]
    assert receipt["counters"] == integrated._zero_counters()
    assert len(receipt["input_extractions"]) == 3
    assert all(
        row["repeat_byte_exact"] is True
        and row["repeat_count"] == 2
        for row in receipt["input_extractions"]
    )
    assert all(
        runtime.calls[
            extractor.PUBLIC_FIXTURES[ordinal].input_sha256
        ]
        == 2
        for ordinal in integrated.PUBLIC_ITEM_FIXTURE_ORDINALS
    )
    summary = receipt["arm_summary"]
    assert [row["arm"] for row in summary["predictions"]] == [
        "semantic_only",
        "legacy",
        "flat",
        "full",
    ]
    for candidate in summary["candidate_receipts"]:
        assert candidate["status"] == "complete"
        assert candidate["flat_proposal_set_hash"] == (
            candidate["full_proposal_set_hash"]
        )
        assert candidate["flat_choice_commitment"] is not None
        assert candidate["full_choice_commitment"] is not None
    assert receipt["implementation_closure_sha256"] == (
        contract_v1.semantic_sha256(
            receipt["implementation_closure"]
        )
    )
    assert {
        "integrated_qualification.py",
        "integrated_qualification.service",
        "intrinsic_arms_v2.py",
        "narrative_correspondence.py",
        "unit_mapping_v2.py",
        "v2_closed_choice.py",
    } <= set(receipt["implementation_closure"])
    body = {
        key: child
        for key, child in receipt.items()
        if key != "self_sha256"
    }
    assert receipt["self_sha256"] == (
        contract_v1.semantic_sha256(body)
    )
    output = tmp_path / "output" / integrated.OUTPUT_NAME
    assert output.read_bytes() == _canonical_bytes(receipt)
    assert output.stat().st_mode & 0o777 == 0o600
    nested = _nested_strings(receipt)
    for fixture in extractor.PUBLIC_FIXTURES:
        assert fixture.story_text not in nested


def test_upstream_failure_is_rejected_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    upstream = _upstream_receipt(
        tmp_path / "upstream.safe.json", manifest
    )
    value = json.loads(upstream.read_text("ascii"))
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    body["qualification_passed"] = False
    upstream.write_bytes(
        _canonical_bytes(
            {
                **body,
                "self_sha256": contract_v1.semantic_sha256(
                    body
                ),
            }
        )
    )
    called = False

    def load(**_kwargs: object) -> object:
        nonlocal called
        called = True
        return _Runtime()

    monkeypatch.setattr(
        extractor, "_load_exact_runtime", load
    )
    with pytest.raises(
        integrated.FixedPublicIntegratedQualificationError,
        match="upstream",
    ):
        integrated.run_fixed_public_integrated_qualification(
            model_root=tmp_path / "model",
            manifest=manifest,
            upstream_aggregate_receipt=upstream,
            output_root=tmp_path / "output",
        )
    assert called is False


def test_extraction_repeat_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        integrated.FixedPublicIntegratedQualificationError,
        match="repeat",
    ):
        _run(
            monkeypatch,
            tmp_path,
            runtime=_Runtime(drift=True),
        )
