from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import tempfile

import pytest

from replication_runtime.gscl_narrative_extractor_v1 import (
    contract as contract_module,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    worker as worker_module,
)
from replication_runtime.gscl_narrative_extractor_v1.contract import (
    CLAIM_SCOPE,
    COMPLETION_SCHEMA,
    INPUT_SCHEMA,
    MAXIMUM_LEXICAL_TOKENS,
    MAXIMUM_COMPLETION_TOKENS,
    MAXIMUM_SPAN_COUNT,
    SPAN_CATALOG_CONTRACT_HASH,
    SPAN_CATALOG_SCHEMA,
    WIRE_COMPLETION_SCHEMA,
    ExecutionClosure,
    NarrativeExtractorRuntimeError,
    StoryRequest,
    admit_story_only_pack_qualification_only,
    build_story_span_catalog,
    canonical_json_bytes,
    decode_input_qualification_only,
    decode_multi_batch_manifest,
    decode_private_output,
    encode_input,
    encode_multi_batch_manifest,
    encode_private_output,
    invalid_result,
    load_trusted_story_only_input_pack,
    require_trusted_story_only_pack,
    validate_completion,
    validate_multi_batch_manifest,
    write_private_output_once,
)
from replication_runtime.gscl_narrative_extractor_v1.worker import (
    MODEL_REPOSITORY_ID,
    PROMPT_SHA256,
    QWEN_ARCHITECTURE,
    GeneratedCompletion,
    LocalQwenRuntime,
    StoryRuntimeFailure,
    SYSTEM_PROMPT,
    USER_INSTRUCTION,
    build_model_asset_manifest_qualification_only,
    load_model_asset_manifest,
    process_trusted_pack,
    process_trusted_pack_test_only,
    prompt_messages,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _closure(prefix: str = "synthetic") -> ExecutionClosure:
    return ExecutionClosure(
        prompt_sha256=PROMPT_SHA256,
        parser_closure_sha256=_sha(f"{prefix}.parser"),
        model_asset_manifest_sha256=_sha(f"{prefix}.model"),
        model_runtime_closure_sha256=_sha(f"{prefix}.runtime"),
        target_double_run_receipt_sha256=_sha(
            f"{prefix}.double-run"
        ),
    )


def _completion(
    story: str,
    *,
    first: str,
    verb: str,
    second: str,
) -> str:
    assert first in story and verb in story and second in story
    catalog = build_story_span_catalog(story)

    def span_id(quote: str) -> str:
        matches = [
            str(row["span_id"])
            for row in catalog
            if row["quote"] == quote and row["occurrence"] == 0
        ]
        assert len(matches) == 1
        return matches[0]

    return json.dumps(
        {
            "generators": [
                {
                    "anchor_span_id": span_id(verb),
                    "causal_orientation": "none",
                    "generator_id": "g0",
                    "generator_kind": "relation",
                    "polarity": "positive",
                    "slot_object_ids": ["o0", "o1"],
                    "temporal_orientation": "forward",
                }
            ],
            "objects": [
                {
                    "object_id": "o0",
                    "span_id": span_id(first),
                },
                {
                    "object_id": "o1",
                    "span_id": span_id(second),
                },
            ],
            "schema_version": WIRE_COMPLETION_SCHEMA,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _pack(
    stories: tuple[str, ...],
    *,
    batch_id: str = "synthetic.batch",
    sequence: int = 0,
):
    requests = tuple(
        StoryRequest(index, story)
        for index, story in enumerate(stories)
    )
    raw = encode_input(
        batch_id=batch_id,
        sequence=sequence,
        requests=requests,
    )
    return admit_story_only_pack_qualification_only(raw), raw


class StubRuntime:
    def __init__(
        self,
        by_story: dict[str, GeneratedCompletion | Exception],
    ):
        self.by_story = by_story
        self.calls: list[str] = []

    def generate(self, story_text: str) -> GeneratedCompletion:
        self.calls.append(story_text)
        value = self.by_story[story_text]
        if isinstance(value, Exception):
            raise value
        return value


def _accept_parser(story: str, completion: str) -> object:
    assert story
    assert json.loads(completion)["schema_version"] == COMPLETION_SCHEMA
    return object()


@pytest.mark.parametrize(
    "forbidden",
    [
        "item_id",
        "query",
        "choice_slot",
        "proverb",
        "gold",
        "answer",
        "label",
        "law",
        "motif",
        "cell",
    ],
)
def test_input_rows_cannot_express_sensitive_fields(
    forbidden: str,
) -> None:
    value = {
        "batch_id": "synthetic.batch",
        "requests": [
            {
                "ordinal": 0,
                "story_text": "Aster guides Birch.",
                forbidden: "forbidden",
            }
        ],
        "schema": INPUT_SCHEMA,
        "sequence": 0,
    }
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        decode_input_qualification_only(canonical_json_bytes(value))
    assert error.value.issue_id == "request_fields_invalid"


def test_prompt_is_single_story_and_claim_is_proposal_only() -> None:
    sensitive = {
        "answer",
        "cell",
        "choice",
        "gold",
        "label",
        "law",
        "motif",
        "proverb",
        "query",
    }
    instruction = f"{SYSTEM_PROMPT}\n{USER_INSTRUCTION}".casefold()
    tokens = {
        token.strip(".,:;()[]{}")
        for token in instruction.replace("_", " ").split()
    }
    assert sensitive.isdisjoint(tokens)
    messages = prompt_messages("Aster guides Birch.")
    rendered = canonical_json_bytes(
        list(messages), newline=False
    ).decode()
    assert "ordinal" not in rendered
    assert "candidate" not in rendered.casefold()
    assert "Aster guides Birch." in rendered
    assert CLAIM_SCOPE == "untrusted_grounded_proposals_only"
    assert "..." not in USER_INSTRUCTION
    assert '"quote":' not in USER_INSTRUCTION
    assert '"anchor_quote"' not in USER_INSTRUCTION
    assert PROMPT_SHA256 == _sha(
        canonical_json_bytes(
            {
                "model_repository_id": MODEL_REPOSITORY_ID,
                "span_catalog_contract_sha256": (
                    SPAN_CATALOG_CONTRACT_HASH
                ),
                "system_prompt": SYSTEM_PROMPT,
                "user_instruction": USER_INSTRUCTION,
                "version": "gscl_narrative_prompt_v4",
            },
            newline=False,
        ).decode()
    )


def test_span_catalog_is_unicode_stable_bounded_and_prompt_exact() -> None:
    story = "阿尔法 guides 贝塔 while 阿尔法 supports 伽马."
    first = build_story_span_catalog(story)
    second = build_story_span_catalog(story)
    assert first == second
    assert [row["span_id"] for row in first] == [
        f"s{index:03d}" for index in range(len(first))
    ]
    alpha_rows = [
        row for row in first if row["quote"] == "阿尔法"
    ]
    assert [row["occurrence"] for row in alpha_rows] == [0, 1]
    catalog_json = canonical_json_bytes(
        {
            "schema": SPAN_CATALOG_SCHEMA,
            "spans": list(first),
        },
        newline=False,
    ).decode("ascii")
    assert catalog_json in prompt_messages(story)[1]["content"]

    maximum_story = " ".join(
        f"token{index}" for index in range(MAXIMUM_LEXICAL_TOKENS)
    )
    maximum = build_story_span_catalog(maximum_story)
    assert len(maximum) == 122
    assert len(maximum) <= MAXIMUM_SPAN_COUNT
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        build_story_span_catalog(maximum_story + " overflow")
    assert (
        error.value.issue_id
        == "story_span_catalog_lexical_count_invalid"
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        build_story_span_catalog("... !!!")
    assert (
        error.value.issue_id
        == "story_span_catalog_lexical_count_invalid"
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        build_story_span_catalog("single")
    assert (
        error.value.issue_id
        == "story_span_catalog_bounds_invalid"
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        build_story_span_catalog("x" * 257)
    assert (
        error.value.issue_id
        == "story_span_catalog_token_too_long"
    )


def test_canonical_input_order_duplicate_float_and_integer_bounds() -> None:
    duplicate = (
        b'{"batch_id":"synthetic.batch","requests":['
        b'{"ordinal":0,"ordinal":0,"story_text":"Aster guides Birch."}],'
        b'"schema":"gscl_narrative_extractor_runtime_v1_input_v2",'
        b'"sequence":0}\n'
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        decode_input_qualification_only(duplicate)
    assert error.value.issue_id == "json_duplicate_key"

    floating = canonical_json_bytes(
        {
            "batch_id": "synthetic.batch",
            "requests": [
                {
                    "ordinal": 0.0,
                    "story_text": "Aster guides Birch.",
                }
            ],
            "schema": INPUT_SCHEMA,
            "sequence": 0,
        }
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        decode_input_qualification_only(floating)
    assert error.value.issue_id == "json_float_forbidden"

    huge = (
        b'{"batch_id":"synthetic.batch","requests":['
        b'{"ordinal":0,"story_text":"Aster guides Birch."}],'
        b'"schema":"gscl_narrative_extractor_runtime_v1_input_v2",'
        b'"sequence":10000000000}\n'
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        decode_input_qualification_only(huge)
    assert error.value.issue_id == "json_integer_out_of_bounds"

    reversed_requests = (
        StoryRequest(1, "Cedar follows Dune."),
        StoryRequest(0, "Aster guides Birch."),
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        encode_input(
            batch_id="synthetic.batch",
            sequence=0,
            requests=reversed_requests,
        )
    assert error.value.issue_id == "request_order_not_canonical"

    inert_sensitive_story = (
        "A character says answer and law inside an inert narrative."
    )
    decoded = decode_input_qualification_only(
        encode_input(
            batch_id="synthetic.inert",
            sequence=0,
            requests=(StoryRequest(0, inert_sensitive_story),),
        )
    )
    assert decoded.requests[0].story_text == inert_sensitive_story


def test_formal_api_requires_trusted_pack_and_exact_runtime() -> None:
    story = "Aster guides Birch."
    pack, _ = _pack((story,))
    runtime = StubRuntime(
        {
            story: GeneratedCompletion(
                _completion(
                    story,
                    first="Aster",
                    verb="guides",
                    second="Birch",
                ),
                31,
            )
        }
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        require_trusted_story_only_pack(pack.requests)
    assert error.value.issue_id == "input_pack_not_admitted"
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        process_trusted_pack(pack, runtime=runtime)
    assert error.value.issue_id == "formal_input_pack_not_trusted"

    raw = encode_input(
        batch_id="synthetic.formal",
        sequence=0,
        requests=(StoryRequest(0, story),),
    )
    with tempfile.TemporaryDirectory(
        prefix="gscl-formal-domain-", dir="/tmp"
    ) as directory:
        input_path = Path(directory) / "input.json"
        input_path.write_bytes(raw)
        input_path.chmod(0o600)
        formal_pack = load_trusted_story_only_input_pack(input_path)
        with pytest.raises(NarrativeExtractorRuntimeError) as error:
            process_trusted_pack(formal_pack, runtime=runtime)
        assert error.value.issue_id == "formal_runtime_not_verified"

        forged_exact_runtime = object.__new__(LocalQwenRuntime)
        with pytest.raises(NarrativeExtractorRuntimeError) as error:
            process_trusted_pack(
                formal_pack, runtime=forged_exact_runtime
            )
        assert error.value.issue_id == "formal_runtime_not_verified"
        with pytest.raises(AttributeError):
            forged_exact_runtime.generate = runtime.generate  # type: ignore[method-assign]

    results, closure = process_trusted_pack_test_only(
        pack,
        runtime=runtime,
        narrative_parser=_accept_parser,
        execution_closure=_closure(),
    )
    assert closure == _closure()
    assert results[0]["generation_valid"] is True
    assert "test_only" in process_trusted_pack_test_only.__name__


def test_completion_is_parser_compatible_bounded_and_anchor_bijective() -> None:
    story = "Aster guides Birch."
    raw = _completion(
        story, first="Aster", verb="guides", second="Birch"
    )
    canonical = validate_completion(
        story, raw, narrative_parser=_accept_parser
    )
    assert canonical != raw
    canonical_payload = json.loads(canonical)
    assert canonical_payload["schema_version"] == COMPLETION_SCHEMA
    assert set(canonical_payload) == {
        "generators",
        "mentions",
        "schema_version",
    }

    from assumption_agent.gscl_narrative_correspondence_v1 import (
        NarrativeSource,
        parse_untrusted_generator_completion,
    )

    parsed = parse_untrusted_generator_completion(
        NarrativeSource("runtime.synthetic", story), canonical
    )
    assert len(parsed.mentions) == 3
    assert len(parsed.generators) == 1

    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        validate_completion(
            story, raw, narrative_parser=None
        )
    assert error.value.issue_id == "validator_unavailable"

    duplicate_anchor = json.loads(raw)
    duplicate_anchor["generators"].append(
        {
            **duplicate_anchor["generators"][0],
            "generator_id": "g1",
        }
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        validate_completion(
            story,
            json.dumps(
                duplicate_anchor,
                separators=(",", ":"),
                sort_keys=True,
            ),
            narrative_parser=_accept_parser,
        )
    assert (
        error.value.issue_id
        == "wire_grounded_span_duplicate"
    )


def test_direct_grounding_wire_rejects_authority_and_reference_attacks() -> None:
    story = "Aster guides Birch beside Cedar."
    base = json.loads(
        _completion(
            story,
            first="Aster",
            verb="guides",
            second="Birch",
        )
    )

    extra_field = json.loads(json.dumps(base))
    extra_field["objects"][0]["kind"] = "object"
    duplicate_object = json.loads(json.dumps(base))
    duplicate_object["objects"][1]["object_id"] = "o0"
    dangling_slot = json.loads(json.dumps(base))
    dangling_slot["generators"][0]["slot_object_ids"][1] = "o9"
    duplicate_slot = json.loads(json.dumps(base))
    duplicate_slot["generators"][0]["slot_object_ids"] = ["o0", "o0"]
    anchor_reuses_object = json.loads(json.dumps(base))
    anchor_reuses_object["generators"][0]["anchor_span_id"] = (
        base["objects"][0]["span_id"]
    )
    unknown_anchor = json.loads(json.dumps(base))
    unknown_anchor["generators"][0]["anchor_span_id"] = "s999"
    unknown_object = json.loads(json.dumps(base))
    unknown_object["objects"][0]["span_id"] = "s999"
    literal_placeholder = json.loads(json.dumps(base))
    literal_placeholder["generators"][0]["anchor_quote"] = "..."
    bad_kind = json.loads(json.dumps(base))
    bad_kind["generators"][0]["generator_kind"] = "similar"
    old_abi = {
        "generators": [],
        "mentions": [],
        "schema_version": COMPLETION_SCHEMA,
    }

    for payload, issue_id in (
        (extra_field, "wire_object_fields_invalid"),
        (duplicate_object, "wire_object_id_duplicate"),
        (dangling_slot, "wire_slot_object_ref_invalid"),
        (duplicate_slot, "wire_slot_object_ref_invalid"),
        (anchor_reuses_object, "wire_grounded_span_duplicate"),
        (unknown_anchor, "wire_anchor_span_id_invalid"),
        (unknown_object, "wire_object_span_id_invalid"),
        (literal_placeholder, "wire_generator_fields_invalid"),
        (bad_kind, "wire_generator_kind_invalid"),
        (old_abi, "wire_completion_fields_invalid"),
    ):
        with pytest.raises(NarrativeExtractorRuntimeError) as error:
            validate_completion(
                story,
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                narrative_parser=_accept_parser,
            )
        assert error.value.issue_id == issue_id

    orphan = json.loads(json.dumps(base))
    cedar_span = next(
        row["span_id"]
        for row in build_story_span_catalog(story)
        if row["quote"] == "Cedar" and row["occurrence"] == 0
    )
    orphan["objects"].append(
        {"object_id": "o2", "span_id": cedar_span}
    )
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        validate_completion(
            story,
            json.dumps(
                orphan,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            narrative_parser=_accept_parser,
        )
    assert error.value.issue_id == "wire_object_coverage_invalid"


def test_maximum_typed_topology_normalizes_independently_of_row_order() -> None:
    assert MAXIMUM_COMPLETION_TOKENS == 512
    story = (
        "Aster guides Birch; Cedar supports Dune; "
        "Aster follows Cedar; Birch greets Dune."
    )
    catalog = build_story_span_catalog(story)

    def span_id(quote: str, occurrence: int = 0) -> str:
        return str(
            next(
                row["span_id"]
                for row in catalog
                if row["quote"] == quote
                and row["occurrence"] == occurrence
            )
        )

    payload = {
        "generators": [
            {
                "anchor_span_id": span_id("guides"),
                "causal_orientation": "none",
                "generator_id": "g0",
                "generator_kind": "relation",
                "polarity": "positive",
                "slot_object_ids": ["o0", "o1"],
                "temporal_orientation": "forward",
            },
            {
                "anchor_span_id": span_id("supports"),
                "causal_orientation": "forward",
                "generator_id": "g1",
                "generator_kind": "causal",
                "polarity": "positive",
                "slot_object_ids": ["o2", "o3"],
                "temporal_orientation": "none",
            },
            {
                "anchor_span_id": span_id("follows"),
                "causal_orientation": "none",
                "generator_id": "g2",
                "generator_kind": "temporal",
                "polarity": "neutral",
                "slot_object_ids": ["o0", "o2"],
                "temporal_orientation": "forward",
            },
            {
                "anchor_span_id": span_id("greets"),
                "causal_orientation": "none",
                "generator_id": "g3",
                "generator_kind": "state_change",
                "polarity": "neutral",
                "slot_object_ids": ["o1", "o3"],
                "temporal_orientation": "none",
            },
        ],
        "objects": [
            {"object_id": "o0", "span_id": span_id("Aster")},
            {"object_id": "o1", "span_id": span_id("Birch")},
            {"object_id": "o2", "span_id": span_id("Cedar")},
            {"object_id": "o3", "span_id": span_id("Dune")},
        ],
        "schema_version": WIRE_COMPLETION_SCHEMA,
    }
    reordered = {
        **payload,
        "generators": list(reversed(payload["generators"])),
        "objects": list(reversed(payload["objects"])),
    }
    first = validate_completion(
        story,
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        narrative_parser=_accept_parser,
    )
    second = validate_completion(
        story,
        json.dumps(
            reordered,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        narrative_parser=_accept_parser,
    )
    assert first == second
    normalized = json.loads(first)
    assert len(normalized["mentions"]) == 8
    assert len(normalized["generators"]) == 4
    assert len(first.encode("utf-8")) < 4_096
    from assumption_agent.gscl_narrative_correspondence_v1 import (
        NarrativeSource,
        parse_untrusted_generator_completion,
    )

    parsed = parse_untrusted_generator_completion(
        NarrativeSource("runtime.maximum", story), first
    )
    assert len(parsed.mentions) == 8
    assert len(parsed.generators) == 4


def test_story_order_swap_does_not_change_independent_completions() -> None:
    first = "Aster guides Birch."
    second = "Cedar follows Dune."
    generations = {
        first: GeneratedCompletion(
            _completion(
                first,
                first="Aster",
                verb="guides",
                second="Birch",
            ),
            41,
        ),
        second: GeneratedCompletion(
            _completion(
                second,
                first="Cedar",
                verb="follows",
                second="Dune",
            ),
            43,
        ),
    }
    pack_one, _ = _pack((first, second))
    pack_two, _ = _pack(
        (second, first),
        batch_id="synthetic.swap",
        sequence=1,
    )
    rows_one, _ = process_trusted_pack_test_only(
        pack_one,
        runtime=StubRuntime(generations),
        narrative_parser=_accept_parser,
        execution_closure=_closure(),
    )
    rows_two, _ = process_trusted_pack_test_only(
        pack_two,
        runtime=StubRuntime(generations),
        narrative_parser=_accept_parser,
        execution_closure=_closure(),
    )
    by_story_one = {
        story: rows_one[index]["completion"]
        for index, story in enumerate((first, second))
    }
    by_story_two = {
        story: rows_two[index]["completion"]
        for index, story in enumerate((second, first))
    }
    assert by_story_one == by_story_two


def test_invalid_and_cap_hit_rows_are_contained_without_source_leak() -> None:
    invalid_story = "Secret synthetic phrase remains private."
    cap_story = "Synthetic cap marker."
    valid_story = "Aster guides Birch."
    pack, _ = _pack((invalid_story, cap_story, valid_story))
    runtime = StubRuntime(
        {
            invalid_story: GeneratedCompletion(
                '{"generators":[],"mentions":[],"schema_version":'
                '"gscl.narrative.extraction.v1"}',
                12,
            ),
            cap_story: GeneratedCompletion(
                "{}",
                MAXIMUM_COMPLETION_TOKENS,
                terminated_by_eos=False,
            ),
            valid_story: GeneratedCompletion(
                _completion(
                    valid_story,
                    first="Aster",
                    verb="guides",
                    second="Birch",
                ),
                31,
            ),
        }
    )
    rows, closure = process_trusted_pack_test_only(
        pack,
        runtime=runtime,
        narrative_parser=_accept_parser,
        execution_closure=_closure(),
    )
    assert rows[0] == invalid_result(
        ordinal=0,
        story_commitment=pack.story_commitments[0],
        error_code="COMPLETION_INVALID",
    )
    assert rows[1] == invalid_result(
        ordinal=1,
        story_commitment=pack.story_commitments[1],
        error_code="OUTPUT_TRUNCATED",
    )
    assert rows[2]["generation_valid"] is True
    private = encode_private_output(
        pack=pack, execution_closure=closure, results=rows
    )
    assert invalid_story.encode() not in private
    assert cap_story.encode() not in private
    assert "completion" not in rows[0]
    assert "completion_sha256" not in rows[1]


def test_any_non_eos_generation_is_fail_closed_as_truncated() -> None:
    story = "Aster guides Birch."
    pack, _ = _pack((story,))
    rows, _ = process_trusted_pack_test_only(
        pack,
        runtime=StubRuntime(
            {
                story: GeneratedCompletion(
                    _completion(
                        story,
                        first="Aster",
                        verb="guides",
                        second="Birch",
                    ),
                    31,
                    terminated_by_eos=False,
                )
            }
        ),
        narrative_parser=_accept_parser,
        execution_closure=_closure(),
    )
    assert rows == [
        invalid_result(
            ordinal=0,
            story_commitment=pack.story_commitments[0],
            error_code="OUTPUT_TRUNCATED",
        )
    ]


def test_uncatalogable_story_abstains_before_model_generation() -> None:
    story = " ".join(f"token{index}" for index in range(33))
    pack, _ = _pack((story,))
    runtime = StubRuntime({})
    rows, _ = process_trusted_pack_test_only(
        pack,
        runtime=runtime,
        narrative_parser=_accept_parser,
        execution_closure=_closure(),
    )
    assert runtime.calls == []
    assert rows == [
        invalid_result(
            ordinal=0,
            story_commitment=pack.story_commitments[0],
            error_code="SPAN_CATALOG_UNAVAILABLE",
        )
    ]


def test_output_binds_input_story_prompt_parser_model_and_replay() -> None:
    story = "Aster guides Birch."
    pack, raw_input = _pack((story,))
    closure = _closure()
    wire_completion = _completion(
        story,
        first="Aster",
        verb="guides",
        second="Birch",
    )
    rows, _ = process_trusted_pack_test_only(
        pack,
        runtime=StubRuntime(
            {
                story: GeneratedCompletion(
                    wire_completion,
                    31,
                )
            }
        ),
        narrative_parser=_accept_parser,
        execution_closure=closure,
    )
    first = encode_private_output(
        pack=pack, execution_closure=closure, results=rows
    )
    second = encode_private_output(
        pack=pack, execution_closure=closure, results=rows
    )
    assert first == second
    decoded = decode_private_output(
        first,
        expected_pack=pack,
        expected_execution_closure=closure,
    )
    assert decoded["input_file_sha256"] == hashlib.sha256(
        raw_input
    ).hexdigest()
    assert decoded["execution_closure"] == closure.payload()
    assert decoded["results"][0]["story_commitment"] == (
        pack.story_commitments[0]
    )
    assert decoded["results"][0]["wire_completion_sha256"] == (
        hashlib.sha256(wire_completion.encode("utf-8")).hexdigest()
    )

    other_closure = _closure("other")
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        decode_private_output(
            first,
            expected_pack=pack,
            expected_execution_closure=other_closure,
        )
    assert (
        error.value.issue_id
        == "output_execution_binding_mismatch"
    )


def test_qualification_output_cannot_cross_formal_admission_domain() -> None:
    story = "Aster guides Birch."
    qualification_pack, raw_input = _pack((story,))
    closure = _closure()
    rows, _ = process_trusted_pack_test_only(
        qualification_pack,
        runtime=StubRuntime(
            {
                story: GeneratedCompletion(
                    _completion(
                        story,
                        first="Aster",
                        verb="guides",
                        second="Birch",
                    ),
                    31,
                )
            }
        ),
        narrative_parser=_accept_parser,
        execution_closure=closure,
    )
    qualification_output = encode_private_output(
        pack=qualification_pack,
        execution_closure=closure,
        results=rows,
    )
    with tempfile.TemporaryDirectory(
        prefix="gscl-output-domain-", dir="/tmp"
    ) as directory:
        input_path = Path(directory) / "input.json"
        input_path.write_bytes(raw_input)
        input_path.chmod(0o600)
        formal_pack = load_trusted_story_only_input_pack(input_path)
        assert (
            formal_pack.input_pack_commitment
            == qualification_pack.input_pack_commitment
        )
        assert (
            formal_pack.input_file_sha256
            == qualification_pack.input_file_sha256
        )
        with pytest.raises(
            NarrativeExtractorRuntimeError
        ) as error:
            decode_private_output(
                qualification_output,
                expected_pack=formal_pack,
                expected_execution_closure=closure,
            )
        assert (
            error.value.issue_id
            == "output_input_binding_mismatch"
        )


def test_multi_batch_manifest_binds_every_output() -> None:
    closure = _closure()
    outputs: list[bytes] = []
    for sequence, (batch_id, story) in enumerate(
        (
            ("synthetic.zero", "Aster guides Birch."),
            ("synthetic.one", "Cedar follows Dune."),
        )
    ):
        first, verb, second = story[:-1].split()
        pack, _ = _pack(
            (story,), batch_id=batch_id, sequence=sequence
        )
        rows, _ = process_trusted_pack_test_only(
            pack,
            runtime=StubRuntime(
                {
                    story: GeneratedCompletion(
                        _completion(
                            story,
                            first=first,
                            verb=verb,
                            second=second,
                        ),
                        31,
                    )
                }
            ),
            narrative_parser=_accept_parser,
            execution_closure=closure,
        )
        outputs.append(
            encode_private_output(
                pack=pack,
                execution_closure=closure,
                results=rows,
            )
        )
    manifest = encode_multi_batch_manifest(tuple(reversed(outputs)))
    decoded = decode_multi_batch_manifest(manifest)
    assert decoded["batch_count"] == 2
    assert [row["sequence"] for row in decoded["batches"]] == [
        0,
        1,
    ]
    assert validate_multi_batch_manifest(
        manifest, outputs
    ) == decoded
    tampered = outputs[0][:-2] + b" \n"
    with pytest.raises(NarrativeExtractorRuntimeError):
        validate_multi_batch_manifest(
            manifest, (tampered, outputs[1])
        )


def test_secure_file_custody_and_exclusive_output() -> None:
    story = "Aster guides Birch."
    requests = (StoryRequest(0, story),)
    raw = encode_input(
        batch_id="synthetic.secure",
        sequence=0,
        requests=requests,
    )
    with tempfile.TemporaryDirectory(
        prefix="gscl-narrative-custody-", dir="/tmp"
    ) as directory:
        root = Path(directory)
        assert root.stat().st_mode & 0o777 == 0o700
        input_path = root / "input.json"
        input_path.write_bytes(raw)
        input_path.chmod(0o600)
        pack = load_trusted_story_only_input_pack(input_path)
        assert pack.input_file_sha256 == hashlib.sha256(raw).hexdigest()
        closure = _closure()
        rows, _ = process_trusted_pack_test_only(
            pack,
            runtime=StubRuntime(
                {
                    story: GeneratedCompletion(
                        _completion(
                            story,
                            first="Aster",
                            verb="guides",
                            second="Birch",
                        ),
                        31,
                    )
                }
            ),
            narrative_parser=_accept_parser,
            execution_closure=closure,
        )
        output_path = root / "private.json"
        write_private_output_once(
            output_path,
            pack=pack,
            execution_closure=closure,
            results=rows,
        )
        assert output_path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(NarrativeExtractorRuntimeError) as error:
            write_private_output_once(
                output_path,
                pack=pack,
                execution_closure=closure,
                results=rows,
            )
        assert error.value.issue_id == "output_target_not_fresh"


def test_landlock_authority_opens_exact_parent_without_root_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="gscl-landlock-direct-", dir="/tmp"
    ) as directory:
        root = Path(directory)
        input_path = root / "input.json"
        raw = encode_input(
            batch_id="synthetic.landlock",
            sequence=0,
            requests=(StoryRequest(0, "Aster guides Birch."),),
        )
        input_path.write_bytes(raw)
        input_path.chmod(0o600)
        monkeypatch.setenv(
            contract_module.
            SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY,
            contract_module.SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY,
        )
        real_open = os.open

        def reject_root_open(
            path: object, *args: object, **kwargs: object
        ) -> int:
            if os.fspath(path) == "/":
                raise AssertionError("Landlock child must not open root")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(
            contract_module.os, "open", reject_root_open
        )
        pack = load_trusted_story_only_input_pack(input_path)
        assert pack.input_file_sha256 == hashlib.sha256(raw).hexdigest()
        output_path = root / "direct-output.bin"
        contract_module._write_bytes_once(output_path, b"bound\n")
        assert output_path.read_bytes() == b"bound\n"
        assert output_path.stat().st_mode & 0o777 == 0o600


def test_landlock_authority_retains_leaf_topology_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        contract_module.
        SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY,
        contract_module.SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY,
    )
    with tempfile.TemporaryDirectory(
        prefix="gscl-landlock-topology-", dir="/tmp"
    ) as directory:
        root = Path(directory)
        loose = root / "loose"
        loose.mkdir(mode=0o755)
        # The formal user service runs with UMask=0077.  Make the deliberately
        # invalid leaf mode explicit so this topology test exercises the same
        # condition inside and outside that service contract.
        loose.chmod(0o755)
        assert loose.stat().st_mode & 0o777 == 0o755
        path = loose / "input.json"
        path.write_bytes(b"{}\n")
        path.chmod(0o600)
        with pytest.raises(
            NarrativeExtractorRuntimeError,
            match="trusted_parent_metadata_invalid",
        ):
            contract_module.secure_read_file(path, maximum=1024)

        trusted = root / "trusted"
        trusted.mkdir(mode=0o700)
        target = trusted / "target.json"
        target.write_bytes(b"{}\n")
        target.chmod(0o600)
        link = trusted / "link.json"
        link.symlink_to(target)
        with pytest.raises(
            NarrativeExtractorRuntimeError,
            match="secure_file_unavailable",
        ):
            contract_module.secure_read_file(link, maximum=1024)


def _synthetic_declarations() -> dict[str, object]:
    return {
        "attention_implementation": "sdpa",
        "chat_template_sha256": _sha("synthetic chat template"),
        "context_limit": 32_768,
        "critical_config": dict(QWEN_ARCHITECTURE),
        "loaded_config_sha256": _sha("synthetic loaded config"),
        "model_class": "Qwen2ForCausalLM",
        "special_token_ids": {
            "bos_token_id": None,
            "eos_token_id": 151_645,
            "pad_token_id": 151_643,
        },
        "tokenizer_class": "Qwen2TokenizerFast",
    }


def _synthetic_runtime_requirements() -> dict[str, object]:
    return {
        "attention_implementation": "sdpa",
        "cuda_version": "12.4",
        "cudnn_version": 90_100,
        "gpu_compute_capability": [8, 6],
        "gpu_name": "Synthetic GPU",
        "python_executable_sha256": _sha(
            "synthetic interpreter"
        ),
        "python_implementation": "CPython",
        "python_version": "3.10.0",
        "torch_version": "2.synthetic",
        "torch_distribution_sha256": _sha(
            "synthetic torch distribution"
        ),
        "transformers_version": "4.synthetic",
        "transformers_distribution_sha256": _sha(
            "synthetic transformers distribution"
        ),
    }


def test_runtime_executable_hash_uses_exact_regular_leaf_without_resolve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "python"
    executable.write_bytes(b"synthetic interpreter")
    monkeypatch.setattr(
        worker_module.sys, "executable", str(executable)
    )
    assert worker_module._hash_runtime_executable() == hashlib.sha256(  # noqa: SLF001
        executable.read_bytes()
    ).hexdigest()
    assert ".resolve(" not in inspect.getsource(  # noqa: SLF001
        worker_module._hash_runtime_executable
    )


def test_runtime_executable_hash_rejects_symlink_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-target"
    target.write_bytes(b"synthetic interpreter")
    executable = tmp_path / "python"
    executable.symlink_to(target)
    monkeypatch.setattr(
        worker_module.sys, "executable", str(executable)
    )
    with pytest.raises(
        NarrativeExtractorRuntimeError,
        match="runtime_executable_topology_invalid",
    ):
        worker_module._hash_runtime_executable()  # noqa: SLF001


def test_distribution_closure_hashes_out_of_root_record_entries_and_binds_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_root = tmp_path / "lib" / "python" / "site-packages"
    package_file = site_root / "fixture" / "__init__.py"
    script_file = tmp_path / "bin" / "fixture-tool"
    package_file.parent.mkdir(parents=True)
    script_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"fixture package v1\n")
    script_file.write_bytes(b"fixture script v1\n")

    class Distribution:
        files = (
            PurePosixPath("fixture/__init__.py"),
            PurePosixPath("../../../bin/fixture-tool"),
        )

        @staticmethod
        def locate_file(path: PurePosixPath) -> Path:
            return site_root / Path(path)

    monkeypatch.setattr(
        worker_module.importlib.metadata,
        "distribution",
        lambda _: Distribution(),
    )
    first = worker_module._distribution_closure_sha256(
        "fixture",
        required_module_origins=(package_file,),
    )
    script_file.write_bytes(b"fixture script v2\n")
    second = worker_module._distribution_closure_sha256(
        "fixture",
        required_module_origins=(package_file,),
    )
    assert first != second

    shadow = tmp_path / "shadow" / "fixture.py"
    shadow.parent.mkdir()
    shadow.write_bytes(b"shadow\n")
    with pytest.raises(
        NarrativeExtractorRuntimeError,
        match="runtime_module_origin_not_in_distribution",
    ):
        worker_module._distribution_closure_sha256(
            "fixture",
            required_module_origins=(shadow,),
        )


def test_distribution_closure_binds_exact_duplicate_record_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_root = tmp_path / "site-packages"
    package_file = site_root / "fixture" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"fixture package\n")

    class Distribution:
        files = (
            PurePosixPath("fixture/__init__.py"),
            PurePosixPath("fixture/__init__.py"),
        )

        @staticmethod
        def locate_file(path: PurePosixPath) -> Path:
            return site_root / Path(path)

    monkeypatch.setattr(
        worker_module.importlib.metadata,
        "distribution",
        lambda _: Distribution(),
    )
    duplicate = worker_module._distribution_closure_sha256(
        "fixture",
        required_module_origins=(package_file,),
    )
    Distribution.files = (
        PurePosixPath("fixture/__init__.py"),
    )
    single = worker_module._distribution_closure_sha256(
        "fixture",
        required_module_origins=(package_file,),
    )
    assert duplicate != single


def test_distribution_closure_rejects_alias_and_changed_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_root = tmp_path / "site-packages"
    package_file = site_root / "fixture" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"fixture package\n")

    class Distribution:
        files = (
            PurePosixPath("fixture/../fixture/__init__.py"),
            PurePosixPath("fixture/__init__.py"),
        )

        @staticmethod
        def locate_file(path: PurePosixPath) -> Path:
            return site_root / Path(path)

    monkeypatch.setattr(
        worker_module.importlib.metadata,
        "distribution",
        lambda _: Distribution(),
    )
    with pytest.raises(
        NarrativeExtractorRuntimeError,
        match="runtime_distribution_declared_path_ambiguous",
    ):
        worker_module._distribution_closure_sha256("fixture")

    Distribution.files = (
        PurePosixPath("fixture/__init__.py"),
        PurePosixPath("fixture/__init__.py"),
    )
    original_hash = worker_module._stable_file_hash_from_fd
    reads = 0

    def changed_second_read(
        descriptor: int, *, maximum: int
    ) -> tuple[str, int]:
        nonlocal reads
        reads += 1
        digest, size = original_hash(
            descriptor, maximum=maximum
        )
        if reads == 2:
            digest = _sha(f"{digest}:changed")
        return digest, size

    monkeypatch.setattr(
        worker_module,
        "_stable_file_hash_from_fd",
        changed_second_read,
    )
    with pytest.raises(
        NarrativeExtractorRuntimeError,
        match="runtime_distribution_file_changed",
    ):
        worker_module._distribution_closure_sha256("fixture")


def test_synthetic_model_manifest_binds_complete_tree_and_runtime() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gscl-narrative-model-", dir="/tmp"
    ) as directory:
        root = Path(directory)
        model_root = root / "model"
        model_root.mkdir(mode=0o700)
        files = {
            ".cache/huggingface/download/weights.lock": (
                b"synthetic nested metadata\n"
            ),
            "config.json": b'{"model_type":"qwen2"}\n',
            "tokenizer.json": b'{"synthetic":true}\n',
            "model.safetensors": b"synthetic model bytes",
        }
        for relative, raw in files.items():
            path = model_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            current_parent = path.parent
            while current_parent != model_root:
                current_parent.chmod(0o700)
                current_parent = current_parent.parent
            path.write_bytes(raw)
            path.chmod(0o600)
        manifest_raw = (
            build_model_asset_manifest_qualification_only(
                model_root=model_root,
                declarations=_synthetic_declarations(),
                runtime_requirements=(
                    _synthetic_runtime_requirements()
                ),
            )
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_bytes(manifest_raw)
        manifest_path.chmod(0o600)
        manifest = load_model_asset_manifest(
            manifest_path=manifest_path, model_root=model_root
        )
        assert manifest.declarations == _synthetic_declarations()
        assert manifest.runtime_requirements == (
            _synthetic_runtime_requirements()
        )
        assert [row["path"] for row in manifest.files] == sorted(
            files
        )
        (model_root / "tokenizer.json").write_bytes(
            b'{"synthetic":false}\n'
        )
        with pytest.raises(NarrativeExtractorRuntimeError) as error:
            load_model_asset_manifest(
                manifest_path=manifest_path,
                model_root=model_root,
            )
        assert error.value.issue_id == "model_tree_drifted"


@pytest.mark.parametrize(
    "relative",
    (
        "/absolute",
        "double//separator",
        "trailing/",
        "./dot",
        "parent/../escape",
        "backslash\\component",
        "nul\x00component",
        "line\nbreak",
    ),
)
def test_model_manifest_rejects_noncanonical_relative_paths(
    relative: str,
) -> None:
    with pytest.raises(NarrativeExtractorRuntimeError) as error:
        worker_module._safe_relative_path(relative)
    assert error.value.issue_id == "model_relative_path_invalid"


def test_target_double_run_and_runtime_closure_are_mandatory() -> None:
    source = inspect.getsource(LocalQwenRuntime.__init__)
    assert "first = self.generate(DETERMINISM_CANARY_STORY)" in source
    assert "second = self.generate(DETERMINISM_CANARY_STORY)" in source
    assert "target_double_run_not_exact" in source
    assert "validate_completion(" in source
    assert "target_double_run_completion_invalid" in source
    assert '"wire_schema": WIRE_COMPLETION_SCHEMA' in source
    closure = _closure()
    assert len(closure.target_double_run_receipt_sha256) == 64
