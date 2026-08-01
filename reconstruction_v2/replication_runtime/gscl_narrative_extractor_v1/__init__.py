"""Offline, source-free narrative extraction runtime.

The package deliberately exposes only anonymous ordinals and one inert story
per request.  Dataset identifiers, alternatives, decisions, and structural
doctrine names are outside its representable input language.
"""

from .contract import (
    COMPLETION_SCHEMA,
    INPUT_SCHEMA,
    MAXIMUM_COMPLETION_TOKENS,
    OUTPUT_SCHEMA,
    ExecutionClosure,
    NarrativeExtractorRuntimeError,
    StoryOnlyInputPack,
    StoryRequest,
    admit_story_only_pack_qualification_only,
    canonical_json_bytes,
    decode_input_qualification_only,
    decode_multi_batch_manifest,
    decode_private_output,
    encode_input,
    encode_multi_batch_manifest,
    encode_private_output,
    load_trusted_story_only_input_pack,
    validate_multi_batch_manifest,
)

__all__ = [
    "COMPLETION_SCHEMA",
    "INPUT_SCHEMA",
    "MAXIMUM_COMPLETION_TOKENS",
    "OUTPUT_SCHEMA",
    "ExecutionClosure",
    "NarrativeExtractorRuntimeError",
    "StoryOnlyInputPack",
    "StoryRequest",
    "admit_story_only_pack_qualification_only",
    "canonical_json_bytes",
    "decode_input_qualification_only",
    "decode_multi_batch_manifest",
    "decode_private_output",
    "encode_input",
    "encode_multi_batch_manifest",
    "encode_private_output",
    "load_trusted_story_only_input_pack",
    "validate_multi_batch_manifest",
]
