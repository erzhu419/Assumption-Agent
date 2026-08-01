"""Custody-bound contract for source-free narrative extraction.

The model output is an untrusted grounded *proposal*.  Exact-substring
grounding proves only that quoted bytes occur in the supplied story; it does
not prove semantic truth, relevance, or a preferred interpretation.  This
runtime adds no semantic judge, NLI stage, efficacy threshold, or gate.

Formal execution accepts only :class:`StoryOnlyInputPack`, which can be
admitted from a securely opened canonical input file.  The ordinary byte
decoder and in-memory admission helper are explicitly qualification-only.
They must not be used by a formal supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence


VERSION = "gscl_narrative_extractor_runtime_v1"
INPUT_SCHEMA = f"{VERSION}_input_v2"
OUTPUT_SCHEMA = f"{VERSION}_private_output_v3"
RESULT_SCHEMA = f"{VERSION}_result_v3"
MULTI_BATCH_MANIFEST_SCHEMA = f"{VERSION}_multi_batch_manifest_v2"
COMPLETION_SCHEMA = "gscl.narrative.extraction.v1"
WIRE_COMPLETION_SCHEMA = "gscl.narrative.catalog_selection.v1"
SPAN_CATALOG_SCHEMA = "gscl.narrative.span_catalog.v1"
CLAIM_SCOPE = "untrusted_grounded_proposals_only"
FORMAL_INPUT_ADMISSION_DOMAIN = "formal_secure_file_admission"
QUALIFICATION_INPUT_ADMISSION_DOMAIN = (
    "qualification_in_memory_admission"
)
SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY = (
    "GSCL_SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY_V1"
)
SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY = (
    "97ff3a77c33a3113712a4c11a9fd347902a12b45f76935023d2ac66377936c35"
)

MAXIMUM_STORY_COUNT = 64
MAXIMUM_STORY_BYTES = 128 * 1024
MAXIMUM_INPUT_BYTES = 20 * 1024 * 1024
MAXIMUM_OUTPUT_BYTES = 10 * 1024 * 1024
MAXIMUM_MANIFEST_BYTES = 2 * 1024 * 1024
MAXIMUM_COMPLETION_BYTES = 64 * 1024
MAXIMUM_COMPLETION_TOKENS = 512
MAXIMUM_JSON_DEPTH = 8
MAXIMUM_JSON_NODES = 2_048
MAXIMUM_MENTIONS = 8
MAXIMUM_OBJECT_MENTIONS = 4
MAXIMUM_GENERATORS = 4
MAXIMUM_SLOTS = 4
MAXIMUM_QUOTE_BYTES = 4_096
MAXIMUM_IDENTIFIER_BYTES = 128
MAXIMUM_JSON_INTEGER = 9_999_999_999
MAXIMUM_BATCH_COUNT = 4_096
MAXIMUM_LEXICAL_TOKENS = 32
MAXIMUM_SPAN_WORDS = 4
MAXIMUM_SPAN_COUNT = 128
MAXIMUM_CATALOG_QUOTE_BYTES = 256
MAXIMUM_SPAN_CATALOG_BYTES = 64 * 1024

ERROR_CODES = frozenset(
    {
        "COMPLETION_INVALID",
        "INPUT_TOO_LONG",
        "MODEL_RUNTIME_ERROR",
        "OUTPUT_TOO_LONG",
        "OUTPUT_TRUNCATED",
        "SPAN_CATALOG_UNAVAILABLE",
        "TOKENIZER_RUNTIME_ERROR",
        "VALIDATOR_UNAVAILABLE",
    }
)

_INPUT_KEYS = frozenset(
    {"batch_id", "requests", "schema", "sequence"}
)
_REQUEST_KEYS = frozenset({"ordinal", "story_text"})
_OUTPUT_KEYS = frozenset(
    {
        "batch_id",
        "claim_scope",
        "execution_closure",
        "input_admission_domain",
        "input_file_sha256",
        "input_pack_commitment",
        "results",
        "schema",
        "sequence",
    }
)
_CLOSURE_KEYS = frozenset(
    {
        "model_asset_manifest_sha256",
        "model_runtime_closure_sha256",
        "parser_closure_sha256",
        "prompt_sha256",
        "target_double_run_receipt_sha256",
    }
)
_VALID_RESULT_KEYS = frozenset(
    {
        "completion",
        "completion_sha256",
        "completion_token_count",
        "generation_valid",
        "ordinal",
        "schema",
        "story_commitment",
        "wire_completion_sha256",
    }
)
_INVALID_RESULT_KEYS = frozenset(
    {
        "error_code",
        "generation_valid",
        "ordinal",
        "schema",
        "story_commitment",
    }
)
_CANONICAL_COMPLETION_KEYS = frozenset(
    {"generators", "mentions", "schema_version"}
)
_WIRE_COMPLETION_KEYS = frozenset(
    {"generators", "objects", "schema_version"}
)
_WIRE_OBJECT_KEYS = frozenset(
    {"object_id", "span_id"}
)
_WIRE_GENERATOR_KEYS = frozenset(
    {
        "anchor_span_id",
        "causal_orientation",
        "generator_id",
        "generator_kind",
        "polarity",
        "slot_object_ids",
        "temporal_orientation",
    }
)
_MENTION_KEYS = frozenset(
    {"kind", "mention_id", "occurrence", "quote"}
)
_GENERATOR_KEYS = frozenset(
    {
        "anchor_mention_id",
        "causal_orientation",
        "generator_id",
        "generator_kind",
        "polarity",
        "slot_mention_ids",
        "temporal_orientation",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "batch_count",
        "batches",
        "execution_closure",
        "input_admission_domain",
        "schema",
        "self_sha256",
    }
)
_BATCH_RECEIPT_KEYS = frozenset(
    {
        "batch_id",
        "generation_invalid_count",
        "generation_valid_count",
        "input_admission_domain",
        "input_file_sha256",
        "input_pack_commitment",
        "output_file_sha256",
        "sequence",
        "story_count",
    }
)
_MENTION_KINDS = frozenset({"generator", "object"})
_GENERATOR_KINDS = frozenset(
    {"causal", "relation", "state_change", "temporal"}
)
_POLARITIES = frozenset({"negative", "neutral", "positive"})
_ORIENTATIONS = frozenset({"forward", "none", "reverse"})
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{1,127}\Z")
_BATCH_ID = re.compile(r"[a-z][a-z0-9.-]{1,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPAN_ID = re.compile(r"s[0-9]{3}\Z")
_LEXICAL_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_FORMAL_FILE_PACK_MARKER = object()
_QUALIFICATION_PACK_MARKER = object()


class NarrativeExtractorRuntimeError(RuntimeError):
    """The private extraction boundary failed closed."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    """Encode the one accepted JSON representation."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise NarrativeExtractorRuntimeError(
            "json_value_not_canonical"
        ) from exc
    return raw + (b"\n" if newline else b"")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, newline=False)
    ).hexdigest()


SPAN_CATALOG_CONTRACT = {
    "catalog_schema": SPAN_CATALOG_SCHEMA,
    "enumeration_order": (
        "lexical_start_ascending_then_word_count_ascending"
    ),
    "lexical_token_pattern": _LEXICAL_TOKEN.pattern,
    "maximum_catalog_quote_bytes": MAXIMUM_CATALOG_QUOTE_BYTES,
    "maximum_lexical_tokens": MAXIMUM_LEXICAL_TOKENS,
    "maximum_span_catalog_bytes": MAXIMUM_SPAN_CATALOG_BYTES,
    "maximum_span_count": MAXIMUM_SPAN_COUNT,
    "maximum_span_words": MAXIMUM_SPAN_WORDS,
    "minimum_span_count": 3,
    "occurrence": "zero_based_exact_substring_occurrence",
    "quote": (
        "story_slice_from_first_lexical_token_start_through_last_token_end"
    ),
    "span_id": "s followed by zero-padded three-digit ordinal",
    "unicode_mode": True,
}
SPAN_CATALOG_CONTRACT_HASH = semantic_sha256(
    SPAN_CATALOG_CONTRACT
)


def _bounded_parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 10:
        raise NarrativeExtractorRuntimeError(
            "json_integer_out_of_bounds"
        )
    parsed = int(value)
    if abs(parsed) > MAXIMUM_JSON_INTEGER:
        raise NarrativeExtractorRuntimeError(
            "json_integer_out_of_bounds"
        )
    return parsed


def _reject_float(_: str) -> None:
    raise NarrativeExtractorRuntimeError("json_float_forbidden")


def _reject_constant(_: str) -> None:
    raise NarrativeExtractorRuntimeError("json_constant_forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NarrativeExtractorRuntimeError("json_duplicate_key")
        result[key] = value
    return result


def _bounded_tree(value: object) -> None:
    nodes = 0

    def visit(node: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAXIMUM_JSON_NODES:
            raise NarrativeExtractorRuntimeError(
                "json_node_count_exceeded"
            )
        if depth > MAXIMUM_JSON_DEPTH:
            raise NarrativeExtractorRuntimeError(
                "json_depth_exceeded"
            )
        if node is None or type(node) in {bool, int, str}:
            return
        if type(node) is list:
            for child in node:
                visit(child, depth + 1)
            return
        if type(node) is dict:
            for key, child in node.items():
                if not isinstance(key, str):
                    raise NarrativeExtractorRuntimeError(
                        "json_key_not_text"
                    )
                visit(child, depth + 1)
            return
        raise NarrativeExtractorRuntimeError("json_type_forbidden")

    visit(value, 0)


def _decode_json(raw: bytes, *, maximum: int, canonical: bool) -> object:
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= maximum:
        raise NarrativeExtractorRuntimeError("json_size_invalid")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise NarrativeExtractorRuntimeError(
            "json_encoding_invalid"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_bounded_parse_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except NarrativeExtractorRuntimeError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise NarrativeExtractorRuntimeError(
            "json_syntax_invalid"
        ) from exc
    _bounded_tree(value)
    if canonical and canonical_json_bytes(value) != raw:
        raise NarrativeExtractorRuntimeError("json_not_canonical")
    return value


def _exact_dict(
    value: object, keys: frozenset[str], issue_id: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise NarrativeExtractorRuntimeError(issue_id)
    return value


def _integer(
    value: object, *, minimum: int, maximum: int, issue_id: str
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise NarrativeExtractorRuntimeError(issue_id)
    return value


def _text(
    value: object,
    *,
    maximum_bytes: int,
    issue_id: str,
    nonempty: bool = True,
) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise NarrativeExtractorRuntimeError(issue_id)
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise NarrativeExtractorRuntimeError(issue_id) from exc
    if size > maximum_bytes or (nonempty and not value.strip()):
        raise NarrativeExtractorRuntimeError(issue_id)
    return value


def _sha256(value: object, issue_id: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NarrativeExtractorRuntimeError(issue_id)
    return value


def _identifier(value: object, issue_id: str) -> str:
    text = _text(
        value,
        maximum_bytes=MAXIMUM_IDENTIFIER_BYTES,
        issue_id=issue_id,
    )
    if _IDENTIFIER.fullmatch(text) is None:
        raise NarrativeExtractorRuntimeError(issue_id)
    return text


def _batch_id(value: object) -> str:
    if not isinstance(value, str) or _BATCH_ID.fullmatch(value) is None:
        raise NarrativeExtractorRuntimeError("batch_id_invalid")
    return value


@dataclass(frozen=True, slots=True)
class StoryRequest:
    """One anonymous and independently generated story."""

    ordinal: int
    story_text: str

    def __post_init__(self) -> None:
        _integer(
            self.ordinal,
            minimum=0,
            maximum=MAXIMUM_STORY_COUNT - 1,
            issue_id="request_ordinal_invalid",
        )
        _text(
            self.story_text,
            maximum_bytes=MAXIMUM_STORY_BYTES,
            issue_id="story_text_invalid",
        )

    def payload(self) -> dict[str, object]:
        return {"ordinal": self.ordinal, "story_text": self.story_text}


@dataclass(frozen=True, slots=True)
class QualificationDecodedInput:
    """Decoded bytes that are not, by themselves, formal custody."""

    batch_id: str
    sequence: int
    requests: tuple[StoryRequest, ...]


@dataclass(frozen=True, slots=True)
class StoryOnlyInputPack:
    """A canonical input whose bytes have crossed the custody boundary."""

    batch_id: str
    sequence: int
    input_file_sha256: str
    input_pack_commitment: str
    requests: tuple[StoryRequest, ...]
    story_commitments: tuple[str, ...]
    _marker: object

    def __post_init__(self) -> None:
        if (
            self._marker is not _FORMAL_FILE_PACK_MARKER
            and self._marker is not _QUALIFICATION_PACK_MARKER
        ):
            raise NarrativeExtractorRuntimeError(
                "input_pack_not_admitted"
            )
        _batch_id(self.batch_id)
        _integer(
            self.sequence,
            minimum=0,
            maximum=MAXIMUM_JSON_INTEGER,
            issue_id="batch_sequence_invalid",
        )
        _sha256(
            self.input_file_sha256, "input_file_sha256_invalid"
        )
        _sha256(
            self.input_pack_commitment,
            "input_pack_commitment_invalid",
        )
        validate_requests(list(self.requests))
        if (
            not isinstance(self.story_commitments, tuple)
            or len(self.story_commitments) != len(self.requests)
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.story_commitments
            )
        ):
            raise NarrativeExtractorRuntimeError(
                "story_commitments_invalid"
            )
        expected = _story_commitments(
            self.batch_id, self.sequence, self.requests
        )
        if expected != self.story_commitments:
            raise NarrativeExtractorRuntimeError(
                "story_commitments_mismatch"
            )
        if (
            _input_pack_commitment(
                batch_id=self.batch_id,
                sequence=self.sequence,
                input_file_sha256=self.input_file_sha256,
                story_commitments=self.story_commitments,
            )
            != self.input_pack_commitment
        ):
            raise NarrativeExtractorRuntimeError(
                "input_pack_commitment_mismatch"
            )

    @property
    def admission_domain(self) -> str:
        if self._marker is _FORMAL_FILE_PACK_MARKER:
            return FORMAL_INPUT_ADMISSION_DOMAIN
        if self._marker is _QUALIFICATION_PACK_MARKER:
            return QUALIFICATION_INPUT_ADMISSION_DOMAIN
        raise NarrativeExtractorRuntimeError(
            "input_admission_domain_invalid"
        )


@dataclass(frozen=True, slots=True)
class ExecutionClosure:
    """All non-story code, prompt, model, and target bindings."""

    prompt_sha256: str
    parser_closure_sha256: str
    model_asset_manifest_sha256: str
    model_runtime_closure_sha256: str
    target_double_run_receipt_sha256: str

    def __post_init__(self) -> None:
        for value, issue_id in (
            (self.prompt_sha256, "prompt_sha256_invalid"),
            (
                self.parser_closure_sha256,
                "parser_closure_sha256_invalid",
            ),
            (
                self.model_asset_manifest_sha256,
                "model_asset_manifest_sha256_invalid",
            ),
            (
                self.model_runtime_closure_sha256,
                "model_runtime_closure_sha256_invalid",
            ),
            (
                self.target_double_run_receipt_sha256,
                "target_double_run_receipt_sha256_invalid",
            ),
        ):
            _sha256(value, issue_id)

    def payload(self) -> dict[str, str]:
        return {
            "model_asset_manifest_sha256": (
                self.model_asset_manifest_sha256
            ),
            "model_runtime_closure_sha256": (
                self.model_runtime_closure_sha256
            ),
            "parser_closure_sha256": self.parser_closure_sha256,
            "prompt_sha256": self.prompt_sha256,
            "target_double_run_receipt_sha256": (
                self.target_double_run_receipt_sha256
            ),
        }

    @classmethod
    def parse(cls, value: object) -> "ExecutionClosure":
        row = _exact_dict(
            value, _CLOSURE_KEYS, "execution_closure_fields_invalid"
        )
        return cls(
            prompt_sha256=row["prompt_sha256"],
            parser_closure_sha256=row["parser_closure_sha256"],
            model_asset_manifest_sha256=(
                row["model_asset_manifest_sha256"]
            ),
            model_runtime_closure_sha256=(
                row["model_runtime_closure_sha256"]
            ),
            target_double_run_receipt_sha256=(
                row["target_double_run_receipt_sha256"]
            ),
        )


def validate_requests(value: object) -> tuple[StoryRequest, ...]:
    if (
        isinstance(value, tuple)
        and all(isinstance(row, StoryRequest) for row in value)
    ):
        requests = value
    else:
        if (
            not isinstance(value, list)
            or not 1 <= len(value) <= MAXIMUM_STORY_COUNT
        ):
            raise NarrativeExtractorRuntimeError(
                "request_count_invalid"
            )
        rows: list[StoryRequest] = []
        for raw in value:
            if isinstance(raw, StoryRequest):
                rows.append(raw)
                continue
            row = _exact_dict(
                raw, _REQUEST_KEYS, "request_fields_invalid"
            )
            rows.append(
                StoryRequest(
                    ordinal=row["ordinal"],
                    story_text=row["story_text"],
                )
            )
        requests = tuple(rows)
    if not 1 <= len(requests) <= MAXIMUM_STORY_COUNT:
        raise NarrativeExtractorRuntimeError("request_count_invalid")
    if tuple(row.ordinal for row in requests) != tuple(
        range(len(requests))
    ):
        raise NarrativeExtractorRuntimeError(
            "request_order_not_canonical"
        )
    return tuple(requests)


def encode_input(
    *,
    batch_id: str,
    sequence: int,
    requests: Sequence[StoryRequest],
) -> bytes:
    batch = _batch_id(batch_id)
    order = _integer(
        sequence,
        minimum=0,
        maximum=MAXIMUM_JSON_INTEGER,
        issue_id="batch_sequence_invalid",
    )
    checked = validate_requests(list(requests))
    raw = canonical_json_bytes(
        {
            "batch_id": batch,
            "requests": [row.payload() for row in checked],
            "schema": INPUT_SCHEMA,
            "sequence": order,
        }
    )
    if len(raw) > MAXIMUM_INPUT_BYTES:
        raise NarrativeExtractorRuntimeError("input_size_invalid")
    return raw


def decode_input_qualification_only(
    raw: bytes,
) -> QualificationDecodedInput:
    """Decode canonical bytes without granting formal custody."""

    value = _decode_json(
        raw, maximum=MAXIMUM_INPUT_BYTES, canonical=True
    )
    envelope = _exact_dict(
        value, _INPUT_KEYS, "input_fields_invalid"
    )
    if envelope["schema"] != INPUT_SCHEMA:
        raise NarrativeExtractorRuntimeError("input_schema_invalid")
    return QualificationDecodedInput(
        batch_id=_batch_id(envelope["batch_id"]),
        sequence=_integer(
            envelope["sequence"],
            minimum=0,
            maximum=MAXIMUM_JSON_INTEGER,
            issue_id="batch_sequence_invalid",
        ),
        requests=validate_requests(envelope["requests"]),
    )


def _story_commitments(
    batch_id: str,
    sequence: int,
    requests: Sequence[StoryRequest],
) -> tuple[str, ...]:
    return tuple(
        semantic_sha256(
            {
                "batch_id": batch_id,
                "ordinal": row.ordinal,
                "sequence": sequence,
                "story_text": row.story_text,
            }
        )
        for row in requests
    )


def _input_pack_commitment(
    *,
    batch_id: str,
    sequence: int,
    input_file_sha256: str,
    story_commitments: Sequence[str],
) -> str:
    return semantic_sha256(
        {
            "batch_id": batch_id,
            "input_file_sha256": input_file_sha256,
            "sequence": sequence,
            "story_commitments": list(story_commitments),
        }
    )


def _admit_decoded(
    decoded: QualificationDecodedInput,
    input_file_sha256: str,
    *,
    marker: object,
) -> StoryOnlyInputPack:
    if (
        marker is not _FORMAL_FILE_PACK_MARKER
        and marker is not _QUALIFICATION_PACK_MARKER
    ):
        raise NarrativeExtractorRuntimeError(
            "input_admission_domain_invalid"
        )
    digest = _sha256(
        input_file_sha256, "input_file_sha256_invalid"
    )
    commitments = _story_commitments(
        decoded.batch_id, decoded.sequence, decoded.requests
    )
    return StoryOnlyInputPack(
        batch_id=decoded.batch_id,
        sequence=decoded.sequence,
        input_file_sha256=digest,
        input_pack_commitment=_input_pack_commitment(
            batch_id=decoded.batch_id,
            sequence=decoded.sequence,
            input_file_sha256=digest,
            story_commitments=commitments,
        ),
        requests=decoded.requests,
        story_commitments=commitments,
        _marker=marker,
    )


def admit_story_only_pack_qualification_only(
    raw: bytes,
) -> StoryOnlyInputPack:
    """Test/qualification helper; formal supervisors must use file admission."""

    return _admit_decoded(
        decode_input_qualification_only(raw),
        hashlib.sha256(raw).hexdigest(),
        marker=_QUALIFICATION_PACK_MARKER,
    )


def require_trusted_story_only_pack(
    value: object,
) -> StoryOnlyInputPack:
    """Require either admitted domain for shared validation utilities."""

    if (
        not isinstance(value, StoryOnlyInputPack)
        or (
            value._marker is not _FORMAL_FILE_PACK_MARKER
            and value._marker is not _QUALIFICATION_PACK_MARKER
        )
    ):
        raise NarrativeExtractorRuntimeError(
            "input_pack_not_admitted"
        )
    value.__post_init__()
    return value


def require_formal_story_only_pack(
    value: object,
) -> StoryOnlyInputPack:
    """Require custody from the secure-file admission path specifically."""

    if (
        not isinstance(value, StoryOnlyInputPack)
        or value._marker is not _FORMAL_FILE_PACK_MARKER
    ):
        raise NarrativeExtractorRuntimeError(
            "formal_input_pack_not_trusted"
        )
    value.__post_init__()
    return value


def _absolute_path(path: Path) -> Path:
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise NarrativeExtractorRuntimeError(
            "path_invalid"
        ) from exc
    if (
        not absolute.is_absolute()
        or absolute.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in absolute.parts[1:])
    ):
        raise NarrativeExtractorRuntimeError("path_invalid")
    return absolute


def _open_trusted_directory(
    path: Path,
    *,
    final_mode: int | None,
    final_owner_current: bool,
) -> tuple[Path, int]:
    """Open each component with openat/no-follow and retain the leaf dirfd."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise NarrativeExtractorRuntimeError("nofollow_unavailable")
    absolute = _absolute_path(path)
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    if (
        os.environ.get(
            SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY
        )
        == SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY
    ):
        # The trusted supervisor resolves and attests every allowed root before
        # applying Landlock, then proves that the private label and linkage
        # packs are unreadable.  Once that sandbox is active, opening "/" to
        # repeat the ancestor walk is intentionally denied.  Open only the
        # exact leaf directory in this narrowly authorised child mode; retain
        # O_NOFOLLOW and the same final-owner/mode checks.  Formal outputs are
        # accepted only alongside the supervisor's Landlock receipt, so merely
        # spoofing this environment value outside that boundary conveys no
        # formal authority.
        try:
            descriptor = os.open(os.fspath(absolute), flags)
        except OSError as exc:
            raise NarrativeExtractorRuntimeError(
                "trusted_path_component_invalid"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            mode = stat.S_IMODE(metadata.st_mode)
            sticky_shared = bool(
                mode & stat.S_ISVTX
                and metadata.st_uid == 0
                and mode & 0o002
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid not in {0, os.getuid()}
                or (
                    mode & 0o022
                    and not sticky_shared
                    and not (
                        final_mode is not None
                        and mode == final_mode
                    )
                )
            ):
                raise NarrativeExtractorRuntimeError(
                    "trusted_path_component_metadata_invalid"
                )
            if (
                (final_mode is not None and mode != final_mode)
                or (
                    final_owner_current
                    and metadata.st_uid != os.getuid()
                )
            ):
                raise NarrativeExtractorRuntimeError(
                    "trusted_parent_metadata_invalid"
                )
            return absolute, descriptor
        except Exception:
            os.close(descriptor)
            raise
    descriptor = os.open("/", flags)
    try:
        for index, component in enumerate(absolute.parts[1:]):
            try:
                child = os.open(
                    component, flags, dir_fd=descriptor
                )
            except OSError as exc:
                raise NarrativeExtractorRuntimeError(
                    "trusted_path_component_invalid"
                ) from exc
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            mode = stat.S_IMODE(metadata.st_mode)
            is_final = index == len(absolute.parts[1:]) - 1
            sticky_shared = bool(
                mode & stat.S_ISVTX
                and metadata.st_uid == 0
                and mode & 0o002
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid not in {0, os.getuid()}
                or (
                    mode & 0o022
                    and not sticky_shared
                    and not (
                        is_final
                        and final_mode is not None
                        and mode == final_mode
                    )
                )
            ):
                raise NarrativeExtractorRuntimeError(
                    "trusted_path_component_metadata_invalid"
                )
            if is_final and (
                (final_mode is not None and mode != final_mode)
                or (
                    final_owner_current
                    and metadata.st_uid != os.getuid()
                )
            ):
                raise NarrativeExtractorRuntimeError(
                    "trusted_parent_metadata_invalid"
                )
        return absolute, descriptor
    except Exception:
        os.close(descriptor)
        raise


@dataclass(frozen=True, slots=True)
class SecureRead:
    raw: bytes
    sha256: str
    size: int
    mtime_ns: int
    ctime_ns: int


def secure_read_file(
    path: Path,
    *,
    maximum: int,
    require_parent_0700: bool = True,
    require_file_0600: bool = True,
) -> SecureRead:
    """Read a stable owned regular file through a trusted parent dirfd."""

    absolute, parent_descriptor = _open_trusted_directory(
        _absolute_path(path).parent,
        final_mode=0o700 if require_parent_0700 else None,
        final_owner_current=True,
    )
    del absolute
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                _absolute_path(path).name,
                flags,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise NarrativeExtractorRuntimeError(
                "secure_file_unavailable"
            ) from exc
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or (
                require_file_0600
                and stat.S_IMODE(before.st_mode) != 0o600
            )
            or not 1 <= before.st_size <= maximum
        ):
            raise NarrativeExtractorRuntimeError(
                "secure_file_metadata_invalid"
            )
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(
                descriptor, min(remaining, 1024 * 1024)
            )
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        binding_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            stat.S_IMODE(before.st_mode),
        )
        binding_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            stat.S_IMODE(after.st_mode),
        )
        if (
            len(raw) > maximum
            or len(raw) != before.st_size
            or binding_before != binding_after
        ):
            raise NarrativeExtractorRuntimeError(
                "secure_file_changed"
            )
        return SecureRead(
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def load_trusted_story_only_input_pack(
    path: Path,
) -> StoryOnlyInputPack:
    read = secure_read_file(path, maximum=MAXIMUM_INPUT_BYTES)
    decoded = decode_input_qualification_only(read.raw)
    return _admit_decoded(
        decoded,
        read.sha256,
        marker=_FORMAL_FILE_PACK_MARKER,
    )


def _completion_json_object(completion: str) -> dict[str, object]:
    text = _text(
        completion,
        maximum_bytes=MAXIMUM_COMPLETION_BYTES,
        issue_id="completion_text_invalid",
    )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_bounded_parse_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except NarrativeExtractorRuntimeError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise NarrativeExtractorRuntimeError(
            "completion_json_invalid"
        ) from exc
    _bounded_tree(value)
    if type(value) is not dict:
        raise NarrativeExtractorRuntimeError(
            "completion_root_invalid"
        )
    return value


def _canonical_completion_json(
    completion: str,
) -> dict[str, object]:
    value = _completion_json_object(completion)
    return _exact_dict(
        value,
        _CANONICAL_COMPLETION_KEYS,
        "completion_fields_invalid",
    )


def _wire_completion_json(completion: str) -> dict[str, object]:
    value = _completion_json_object(completion)
    return _exact_dict(
        value,
        _WIRE_COMPLETION_KEYS,
        "wire_completion_fields_invalid",
    )


def _occurrence_start(
    story_text: str, quote: str, occurrence: int
) -> int:
    starts: list[int] = []
    offset = 0
    while True:
        start = story_text.find(quote, offset)
        if start < 0:
            break
        starts.append(start)
        offset = start + 1
    if occurrence >= len(starts):
        raise NarrativeExtractorRuntimeError(
            "completion_quote_not_grounded"
        )
    return starts[occurrence]


def build_story_span_catalog(
    story_text: str,
) -> tuple[dict[str, object], ...]:
    """Enumerate the only grounded spans expressible by the wire grammar."""

    story = _text(
        story_text,
        maximum_bytes=MAXIMUM_STORY_BYTES,
        issue_id="story_text_invalid",
    )
    tokens = tuple(_LEXICAL_TOKEN.finditer(story))
    if not 1 <= len(tokens) <= MAXIMUM_LEXICAL_TOKENS:
        raise NarrativeExtractorRuntimeError(
            "story_span_catalog_lexical_count_invalid"
        )
    for token in tokens:
        if (
            len(
                story[token.start() : token.end()].encode(
                    "utf-8", errors="strict"
                )
            )
            > MAXIMUM_CATALOG_QUOTE_BYTES
        ):
            raise NarrativeExtractorRuntimeError(
                "story_span_catalog_token_too_long"
            )

    rows: list[dict[str, object]] = []
    for start_index, first in enumerate(tokens):
        for word_count in range(1, MAXIMUM_SPAN_WORDS + 1):
            end_index = start_index + word_count
            if end_index > len(tokens):
                break
            last = tokens[end_index - 1]
            quote = story[first.start() : last.end()]
            if (
                len(quote.encode("utf-8", errors="strict"))
                > MAXIMUM_CATALOG_QUOTE_BYTES
            ):
                break
            positions: list[int] = []
            offset = 0
            while True:
                position = story.find(quote, offset)
                if position < 0:
                    break
                positions.append(position)
                offset = position + 1
            try:
                occurrence = positions.index(first.start())
            except ValueError as exc:
                raise NarrativeExtractorRuntimeError(
                    "story_span_catalog_grounding_invalid"
                ) from exc
            rows.append(
                {
                    "occurrence": occurrence,
                    "quote": quote,
                    "span_id": f"s{len(rows):03d}",
                }
            )
    if (
        len(rows) < 3
        or len(rows) > MAXIMUM_SPAN_COUNT
        or len(
            canonical_json_bytes(
                {
                    "schema": SPAN_CATALOG_SCHEMA,
                    "spans": rows,
                },
                newline=False,
            )
        )
        > MAXIMUM_SPAN_CATALOG_BYTES
    ):
        raise NarrativeExtractorRuntimeError(
            "story_span_catalog_bounds_invalid"
        )
    return tuple(rows)


NarrativeParser = Callable[[str, str], object]


def validate_completion(
    story_text: str,
    completion: str,
    *,
    narrative_parser: NarrativeParser,
) -> str:
    """Validate direct grounding and normalize it to the canonical parser ABI."""

    if narrative_parser is None or not callable(narrative_parser):
        raise NarrativeExtractorRuntimeError(
            "validator_unavailable"
        )
    story = _text(
        story_text,
        maximum_bytes=MAXIMUM_STORY_BYTES,
        issue_id="story_text_invalid",
    )
    value = _wire_completion_json(completion)
    if value["schema_version"] != WIRE_COMPLETION_SCHEMA:
        raise NarrativeExtractorRuntimeError(
            "wire_completion_schema_invalid"
        )
    catalog = build_story_span_catalog(story)
    catalog_by_id = {
        str(row["span_id"]): row for row in catalog
    }
    raw_objects = value["objects"]
    raw_generators = value["generators"]
    if (
        type(raw_objects) is not list
        or not 2 <= len(raw_objects) <= MAXIMUM_OBJECT_MENTIONS
    ):
        raise NarrativeExtractorRuntimeError(
            "wire_object_count_invalid"
        )
    if (
        type(raw_generators) is not list
        or not 1 <= len(raw_generators) <= MAXIMUM_GENERATORS
    ):
        raise NarrativeExtractorRuntimeError(
            "wire_generator_count_invalid"
        )

    objects: dict[str, dict[str, object]] = {}
    spans: set[tuple[int, int]] = set()
    for raw in raw_objects:
        row = _exact_dict(
            raw,
            _WIRE_OBJECT_KEYS,
            "wire_object_fields_invalid",
        )
        object_id = _identifier(
            row["object_id"], "wire_object_id_invalid"
        )
        if object_id in objects:
            raise NarrativeExtractorRuntimeError(
                "wire_object_id_duplicate"
            )
        span_id = row["span_id"]
        if (
            not isinstance(span_id, str)
            or _SPAN_ID.fullmatch(span_id) is None
            or span_id not in catalog_by_id
        ):
            raise NarrativeExtractorRuntimeError(
                "wire_object_span_id_invalid"
            )
        catalog_row = catalog_by_id[span_id]
        quote = str(catalog_row["quote"])
        occurrence = int(catalog_row["occurrence"])
        start = _occurrence_start(story, quote, occurrence)
        span = (start, start + len(quote))
        if span in spans:
            raise NarrativeExtractorRuntimeError(
                "wire_grounded_span_duplicate"
            )
        spans.add(span)
        objects[object_id] = {
            "object_id": object_id,
            "occurrence": occurrence,
            "quote": quote,
            "span_id": span_id,
            "span": span,
        }

    generator_ids: set[str] = set()
    used_objects: set[str] = set()
    generators: list[dict[str, object]] = []
    for raw in raw_generators:
        row = _exact_dict(
            raw,
            _WIRE_GENERATOR_KEYS,
            "wire_generator_fields_invalid",
        )
        generator_id = _identifier(
            row["generator_id"],
            "wire_generator_id_invalid",
        )
        if generator_id in generator_ids:
            raise NarrativeExtractorRuntimeError(
                "wire_generator_id_duplicate"
            )
        generator_ids.add(generator_id)
        anchor_span_id = row["anchor_span_id"]
        if (
            not isinstance(anchor_span_id, str)
            or _SPAN_ID.fullmatch(anchor_span_id) is None
            or anchor_span_id not in catalog_by_id
        ):
            raise NarrativeExtractorRuntimeError(
                "wire_anchor_span_id_invalid"
            )
        anchor_catalog_row = catalog_by_id[anchor_span_id]
        anchor_quote = str(anchor_catalog_row["quote"])
        anchor_occurrence = int(
            anchor_catalog_row["occurrence"]
        )
        anchor_start = _occurrence_start(
            story, anchor_quote, anchor_occurrence
        )
        anchor_span = (
            anchor_start,
            anchor_start + len(anchor_quote),
        )
        if anchor_span in spans:
            raise NarrativeExtractorRuntimeError(
                "wire_grounded_span_duplicate"
            )
        spans.add(anchor_span)
        slots = row["slot_object_ids"]
        if (
            type(slots) is not list
            or not 2 <= len(slots) <= MAXIMUM_SLOTS
        ):
            raise NarrativeExtractorRuntimeError(
                "wire_slot_count_invalid"
            )
        checked_slots = [
            _identifier(slot, "wire_slot_object_id_invalid")
            for slot in slots
        ]
        if (
            len(set(checked_slots)) != len(checked_slots)
            or any(
                slot not in objects
                for slot in checked_slots
            )
        ):
            raise NarrativeExtractorRuntimeError(
                "wire_slot_object_ref_invalid"
            )
        used_objects.update(checked_slots)
        if row["generator_kind"] not in _GENERATOR_KINDS:
            raise NarrativeExtractorRuntimeError(
                "wire_generator_kind_invalid"
            )
        if row["polarity"] not in _POLARITIES:
            raise NarrativeExtractorRuntimeError(
                "wire_polarity_invalid"
            )
        if (
            row["temporal_orientation"] not in _ORIENTATIONS
            or row["causal_orientation"] not in _ORIENTATIONS
        ):
            raise NarrativeExtractorRuntimeError(
                "wire_orientation_invalid"
            )
        generators.append(
            {
                "anchor_occurrence": anchor_occurrence,
                "anchor_quote": anchor_quote,
                "anchor_span_id": anchor_span_id,
                "anchor_span": anchor_span,
                "causal_orientation": row[
                    "causal_orientation"
                ],
                "generator_id": generator_id,
                "generator_kind": row["generator_kind"],
                "polarity": row["polarity"],
                "slot_object_ids": checked_slots,
                "temporal_orientation": row[
                    "temporal_orientation"
                ],
            }
        )
    if used_objects != set(objects):
        raise NarrativeExtractorRuntimeError(
            "wire_object_coverage_invalid"
        )

    ordered_generators = sorted(
        generators,
        key=lambda row: (
            row["anchor_span"],
            row["generator_id"],
        ),
    )
    reserved_ids = set(objects)
    next_anchor = 0
    anchor_ids: dict[str, str] = {}
    for row in ordered_generators:
        while f"a{next_anchor}" in reserved_ids:
            next_anchor += 1
        anchor_id = f"a{next_anchor}"
        next_anchor += 1
        reserved_ids.add(anchor_id)
        anchor_ids[str(row["generator_id"])] = anchor_id

    mentions: list[dict[str, object]] = []
    mention_sort_keys: dict[str, tuple[int, int, str, str]] = {}
    for object_id, row in objects.items():
        span = row["span"]
        mentions.append(
            {
                "kind": "object",
                "mention_id": object_id,
                "occurrence": row["occurrence"],
                "quote": row["quote"],
            }
        )
        mention_sort_keys[object_id] = (
            int(span[0]),
            int(span[1]),
            "object",
            object_id,
        )
    for row in ordered_generators:
        generator_id = str(row["generator_id"])
        anchor_id = anchor_ids[generator_id]
        span = row["anchor_span"]
        mentions.append(
            {
                "kind": "generator",
                "mention_id": anchor_id,
                "occurrence": row["anchor_occurrence"],
                "quote": row["anchor_quote"],
            }
        )
        mention_sort_keys[anchor_id] = (
            int(span[0]),
            int(span[1]),
            "generator",
            anchor_id,
        )
    mentions.sort(
        key=lambda row: mention_sort_keys[str(row["mention_id"])]
    )
    canonical_generators = [
        {
            "anchor_mention_id": anchor_ids[
                str(row["generator_id"])
            ],
            "causal_orientation": row["causal_orientation"],
            "generator_id": row["generator_id"],
            "generator_kind": row["generator_kind"],
            "polarity": row["polarity"],
            "slot_mention_ids": row["slot_object_ids"],
            "temporal_orientation": row["temporal_orientation"],
        }
        for row in ordered_generators
    ]

    canonical = canonical_json_bytes(
        {
            "generators": canonical_generators,
            "mentions": mentions,
            "schema_version": COMPLETION_SCHEMA,
        },
        newline=False,
    ).decode("ascii")
    if len(canonical) > MAXIMUM_COMPLETION_BYTES:
        raise NarrativeExtractorRuntimeError(
            "completion_canonical_size_invalid"
        )
    try:
        narrative_parser(story, canonical)
    except NarrativeExtractorRuntimeError:
        raise
    except Exception as exc:
        raise NarrativeExtractorRuntimeError(
            "completion_parser_rejected"
        ) from exc
    return canonical


def valid_result(
    *,
    ordinal: int,
    story_commitment: str,
    completion: str,
    completion_token_count: int,
    wire_completion_sha256: str | None = None,
) -> dict[str, object]:
    ordinal = _integer(
        ordinal,
        minimum=0,
        maximum=MAXIMUM_STORY_COUNT - 1,
        issue_id="result_ordinal_invalid",
    )
    story_digest = _sha256(
        story_commitment, "result_story_commitment_invalid"
    )
    completion = _text(
        completion,
        maximum_bytes=MAXIMUM_COMPLETION_BYTES,
        issue_id="result_completion_invalid",
    )
    completion_value = _canonical_completion_json(completion)
    if (
        canonical_json_bytes(
            completion_value, newline=False
        ).decode("ascii")
        != completion
    ):
        raise NarrativeExtractorRuntimeError(
            "result_completion_not_canonical"
        )
    token_count = _integer(
        completion_token_count,
        minimum=1,
        maximum=MAXIMUM_COMPLETION_TOKENS - 1,
        issue_id="result_token_count_invalid",
    )
    wire_digest = (
        hashlib.sha256(completion.encode("utf-8")).hexdigest()
        if wire_completion_sha256 is None
        else _sha256(
            wire_completion_sha256,
            "result_wire_completion_sha256_invalid",
        )
    )
    return {
        "completion": completion,
        "completion_sha256": hashlib.sha256(
            completion.encode("utf-8")
        ).hexdigest(),
        "completion_token_count": token_count,
        "generation_valid": True,
        "ordinal": ordinal,
        "schema": RESULT_SCHEMA,
        "story_commitment": story_digest,
        "wire_completion_sha256": wire_digest,
    }


def invalid_result(
    *, ordinal: int, story_commitment: str, error_code: str
) -> dict[str, object]:
    ordinal = _integer(
        ordinal,
        minimum=0,
        maximum=MAXIMUM_STORY_COUNT - 1,
        issue_id="result_ordinal_invalid",
    )
    story_digest = _sha256(
        story_commitment, "result_story_commitment_invalid"
    )
    if error_code not in ERROR_CODES:
        raise NarrativeExtractorRuntimeError(
            "result_error_code_invalid"
        )
    return {
        "error_code": error_code,
        "generation_valid": False,
        "ordinal": ordinal,
        "schema": RESULT_SCHEMA,
        "story_commitment": story_digest,
    }


def validate_private_results(
    results: Sequence[Mapping[str, object]],
    *,
    pack: StoryOnlyInputPack,
) -> list[dict[str, object]]:
    trusted = require_trusted_story_only_pack(pack)
    rows = list(results)
    if len(rows) != len(trusted.requests):
        raise NarrativeExtractorRuntimeError("result_count_invalid")
    checked: list[dict[str, object]] = []
    for position, raw in enumerate(rows):
        if type(raw) is not dict:
            raise NarrativeExtractorRuntimeError(
                "result_type_invalid"
            )
        if raw.get("generation_valid") is True:
            row = _exact_dict(
                raw, _VALID_RESULT_KEYS, "valid_result_fields_invalid"
            )
            if row["schema"] != RESULT_SCHEMA:
                raise NarrativeExtractorRuntimeError(
                    "result_schema_invalid"
                )
            completion = _text(
                row["completion"],
                maximum_bytes=MAXIMUM_COMPLETION_BYTES,
                issue_id="result_completion_invalid",
            )
            completion_value = _canonical_completion_json(completion)
            if (
                canonical_json_bytes(
                    completion_value, newline=False
                ).decode("ascii")
                != completion
            ):
                raise NarrativeExtractorRuntimeError(
                    "result_completion_not_canonical"
                )
            digest = row["completion_sha256"]
            wire_digest = row["wire_completion_sha256"]
            _integer(
                row["completion_token_count"],
                minimum=1,
                maximum=MAXIMUM_COMPLETION_TOKENS - 1,
                issue_id="result_token_count_invalid",
            )
            if (
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or hashlib.sha256(
                    completion.encode("utf-8")
                ).hexdigest()
                != digest
            ):
                raise NarrativeExtractorRuntimeError(
                    "result_completion_hash_invalid"
                )
            _sha256(
                wire_digest,
                "result_wire_completion_sha256_invalid",
            )
        elif raw.get("generation_valid") is False:
            row = _exact_dict(
                raw,
                _INVALID_RESULT_KEYS,
                "invalid_result_fields_invalid",
            )
            if (
                row["schema"] != RESULT_SCHEMA
                or row["error_code"] not in ERROR_CODES
            ):
                raise NarrativeExtractorRuntimeError(
                    "invalid_result_value_invalid"
                )
        else:
            raise NarrativeExtractorRuntimeError(
                "result_validity_invalid"
            )
        if (
            row["ordinal"] != position
            or row["story_commitment"]
            != trusted.story_commitments[position]
        ):
            raise NarrativeExtractorRuntimeError(
                "result_story_binding_invalid"
            )
        checked.append(dict(row))
    return checked


def private_output_payload(
    *,
    pack: StoryOnlyInputPack,
    execution_closure: ExecutionClosure,
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    trusted = require_trusted_story_only_pack(pack)
    if not isinstance(execution_closure, ExecutionClosure):
        raise NarrativeExtractorRuntimeError(
            "execution_closure_invalid"
        )
    checked = validate_private_results(results, pack=trusted)
    return {
        "batch_id": trusted.batch_id,
        "claim_scope": CLAIM_SCOPE,
        "execution_closure": execution_closure.payload(),
        "input_admission_domain": trusted.admission_domain,
        "input_file_sha256": trusted.input_file_sha256,
        "input_pack_commitment": trusted.input_pack_commitment,
        "results": checked,
        "schema": OUTPUT_SCHEMA,
        "sequence": trusted.sequence,
    }


def encode_private_output(
    *,
    pack: StoryOnlyInputPack,
    execution_closure: ExecutionClosure,
    results: Sequence[Mapping[str, object]],
) -> bytes:
    raw = canonical_json_bytes(
        private_output_payload(
            pack=pack,
            execution_closure=execution_closure,
            results=results,
        )
    )
    if len(raw) > MAXIMUM_OUTPUT_BYTES:
        raise NarrativeExtractorRuntimeError("output_size_invalid")
    return raw


def _parse_output_shape(raw: bytes) -> dict[str, object]:
    value = _decode_json(
        raw, maximum=MAXIMUM_OUTPUT_BYTES, canonical=True
    )
    envelope = _exact_dict(
        value, _OUTPUT_KEYS, "output_fields_invalid"
    )
    if (
        envelope["schema"] != OUTPUT_SCHEMA
        or envelope["claim_scope"] != CLAIM_SCOPE
    ):
        raise NarrativeExtractorRuntimeError(
            "output_identity_invalid"
        )
    _batch_id(envelope["batch_id"])
    _integer(
        envelope["sequence"],
        minimum=0,
        maximum=MAXIMUM_JSON_INTEGER,
        issue_id="batch_sequence_invalid",
    )
    _sha256(
        envelope["input_file_sha256"],
        "input_file_sha256_invalid",
    )
    if envelope["input_admission_domain"] not in {
        FORMAL_INPUT_ADMISSION_DOMAIN,
        QUALIFICATION_INPUT_ADMISSION_DOMAIN,
    }:
        raise NarrativeExtractorRuntimeError(
            "output_admission_domain_invalid"
        )
    _sha256(
        envelope["input_pack_commitment"],
        "input_pack_commitment_invalid",
    )
    ExecutionClosure.parse(envelope["execution_closure"])
    if (
        type(envelope["results"]) is not list
        or not 1
        <= len(envelope["results"])
        <= MAXIMUM_STORY_COUNT
    ):
        raise NarrativeExtractorRuntimeError(
            "result_count_invalid"
        )
    for position, raw_row in enumerate(envelope["results"]):
        if type(raw_row) is not dict:
            raise NarrativeExtractorRuntimeError(
                "result_type_invalid"
            )
        keys = (
            _VALID_RESULT_KEYS
            if raw_row.get("generation_valid") is True
            else _INVALID_RESULT_KEYS
            if raw_row.get("generation_valid") is False
            else frozenset()
        )
        row = _exact_dict(
            raw_row, keys, "result_fields_invalid"
        )
        if row["ordinal"] != position:
            raise NarrativeExtractorRuntimeError(
                "result_order_invalid"
            )
        _sha256(
            row["story_commitment"],
            "result_story_commitment_invalid",
        )
        if row["schema"] != RESULT_SCHEMA:
            raise NarrativeExtractorRuntimeError(
                "result_schema_invalid"
            )
        if row["generation_valid"] is True:
            completion = _text(
                row["completion"],
                maximum_bytes=MAXIMUM_COMPLETION_BYTES,
                issue_id="result_completion_invalid",
            )
            completion_value = _canonical_completion_json(completion)
            if (
                canonical_json_bytes(
                    completion_value, newline=False
                ).decode("ascii")
                != completion
            ):
                raise NarrativeExtractorRuntimeError(
                    "result_completion_not_canonical"
                )
            if (
                hashlib.sha256(
                    completion.encode("utf-8")
                ).hexdigest()
                != row["completion_sha256"]
            ):
                raise NarrativeExtractorRuntimeError(
                    "result_completion_hash_invalid"
                )
            _sha256(
                row["wire_completion_sha256"],
                "result_wire_completion_sha256_invalid",
            )
            _integer(
                row["completion_token_count"],
                minimum=1,
                maximum=MAXIMUM_COMPLETION_TOKENS - 1,
                issue_id="result_token_count_invalid",
            )
        elif row["error_code"] not in ERROR_CODES:
            raise NarrativeExtractorRuntimeError(
                "result_error_code_invalid"
            )
    return envelope


def decode_private_output(
    raw: bytes,
    *,
    expected_pack: StoryOnlyInputPack | None = None,
    expected_execution_closure: ExecutionClosure | None = None,
) -> dict[str, object]:
    envelope = _parse_output_shape(raw)
    if expected_pack is not None:
        trusted = require_trusted_story_only_pack(expected_pack)
        if (
            envelope["batch_id"] != trusted.batch_id
            or envelope["sequence"] != trusted.sequence
            or envelope["input_admission_domain"]
            != trusted.admission_domain
            or envelope["input_file_sha256"]
            != trusted.input_file_sha256
            or envelope["input_pack_commitment"]
            != trusted.input_pack_commitment
            or [
                row["story_commitment"]
                for row in envelope["results"]
            ]
            != list(trusted.story_commitments)
        ):
            raise NarrativeExtractorRuntimeError(
                "output_input_binding_mismatch"
            )
    if expected_execution_closure is not None and (
        envelope["execution_closure"]
        != expected_execution_closure.payload()
    ):
        raise NarrativeExtractorRuntimeError(
            "output_execution_binding_mismatch"
        )
    return envelope


def _batch_receipt(raw: bytes) -> dict[str, object]:
    output = decode_private_output(raw)
    results = output["results"]
    return {
        "batch_id": output["batch_id"],
        "generation_invalid_count": sum(
            row["generation_valid"] is False for row in results
        ),
        "generation_valid_count": sum(
            row["generation_valid"] is True for row in results
        ),
        "input_file_sha256": output["input_file_sha256"],
        "input_admission_domain": output[
            "input_admission_domain"
        ],
        "input_pack_commitment": output["input_pack_commitment"],
        "output_file_sha256": hashlib.sha256(raw).hexdigest(),
        "sequence": output["sequence"],
        "story_count": len(results),
    }


def encode_multi_batch_manifest(
    output_batches: Sequence[bytes],
) -> bytes:
    if not 1 <= len(output_batches) <= MAXIMUM_BATCH_COUNT:
        raise NarrativeExtractorRuntimeError(
            "manifest_batch_count_invalid"
        )
    parsed = [decode_private_output(raw) for raw in output_batches]
    parsed.sort(key=lambda row: row["sequence"])
    if [row["sequence"] for row in parsed] != list(
        range(len(parsed))
    ):
        raise NarrativeExtractorRuntimeError(
            "manifest_sequence_invalid"
        )
    if len({row["batch_id"] for row in parsed}) != len(parsed):
        raise NarrativeExtractorRuntimeError(
            "manifest_batch_id_duplicate"
        )
    closure = parsed[0]["execution_closure"]
    if any(row["execution_closure"] != closure for row in parsed):
        raise NarrativeExtractorRuntimeError(
            "manifest_execution_closure_mismatch"
        )
    admission_domain = parsed[0]["input_admission_domain"]
    if any(
        row["input_admission_domain"] != admission_domain
        for row in parsed
    ):
        raise NarrativeExtractorRuntimeError(
            "manifest_admission_domain_mismatch"
        )
    raw_by_hash = {
        hashlib.sha256(raw).hexdigest(): raw for raw in output_batches
    }
    receipts = sorted(
        (
            _batch_receipt(raw)
            for raw in raw_by_hash.values()
        ),
        key=lambda row: row["sequence"],
    )
    body = {
        "batch_count": len(receipts),
        "batches": receipts,
        "execution_closure": closure,
        "input_admission_domain": admission_domain,
        "schema": MULTI_BATCH_MANIFEST_SCHEMA,
    }
    raw = canonical_json_bytes(
        {**body, "self_sha256": semantic_sha256(body)}
    )
    if len(raw) > MAXIMUM_MANIFEST_BYTES:
        raise NarrativeExtractorRuntimeError(
            "manifest_size_invalid"
        )
    return raw


def decode_multi_batch_manifest(raw: bytes) -> dict[str, object]:
    value = _decode_json(
        raw, maximum=MAXIMUM_MANIFEST_BYTES, canonical=True
    )
    manifest = _exact_dict(
        value, _MANIFEST_KEYS, "manifest_fields_invalid"
    )
    body = {
        key: value
        for key, value in manifest.items()
        if key != "self_sha256"
    }
    if (
        manifest["schema"] != MULTI_BATCH_MANIFEST_SCHEMA
        or semantic_sha256(body) != manifest["self_sha256"]
    ):
        raise NarrativeExtractorRuntimeError(
            "manifest_identity_invalid"
        )
    ExecutionClosure.parse(manifest["execution_closure"])
    if manifest["input_admission_domain"] not in {
        FORMAL_INPUT_ADMISSION_DOMAIN,
        QUALIFICATION_INPUT_ADMISSION_DOMAIN,
    }:
        raise NarrativeExtractorRuntimeError(
            "manifest_admission_domain_invalid"
        )
    batches = manifest["batches"]
    count = _integer(
        manifest["batch_count"],
        minimum=1,
        maximum=MAXIMUM_BATCH_COUNT,
        issue_id="manifest_batch_count_invalid",
    )
    if type(batches) is not list or len(batches) != count:
        raise NarrativeExtractorRuntimeError(
            "manifest_batch_count_invalid"
        )
    batch_ids: set[str] = set()
    for position, raw_row in enumerate(batches):
        row = _exact_dict(
            raw_row,
            _BATCH_RECEIPT_KEYS,
            "manifest_batch_fields_invalid",
        )
        if (
            row["input_admission_domain"]
            != manifest["input_admission_domain"]
        ):
            raise NarrativeExtractorRuntimeError(
                "manifest_admission_domain_mismatch"
            )
        batch_ids.add(_batch_id(row["batch_id"]))
        if row["sequence"] != position:
            raise NarrativeExtractorRuntimeError(
                "manifest_sequence_invalid"
            )
        story_count = _integer(
            row["story_count"],
            minimum=1,
            maximum=MAXIMUM_STORY_COUNT,
            issue_id="manifest_story_count_invalid",
        )
        valid = _integer(
            row["generation_valid_count"],
            minimum=0,
            maximum=story_count,
            issue_id="manifest_valid_count_invalid",
        )
        invalid = _integer(
            row["generation_invalid_count"],
            minimum=0,
            maximum=story_count,
            issue_id="manifest_invalid_count_invalid",
        )
        if valid + invalid != story_count:
            raise NarrativeExtractorRuntimeError(
                "manifest_result_counts_invalid"
            )
        for field in (
            "input_file_sha256",
            "input_pack_commitment",
            "output_file_sha256",
        ):
            _sha256(row[field], f"manifest_{field}_invalid")
    if len(batch_ids) != count:
        raise NarrativeExtractorRuntimeError(
            "manifest_batch_id_duplicate"
        )
    return manifest


def validate_multi_batch_manifest(
    manifest_raw: bytes, output_batches: Sequence[bytes]
) -> dict[str, object]:
    manifest = decode_multi_batch_manifest(manifest_raw)
    expected = encode_multi_batch_manifest(output_batches)
    if expected != manifest_raw:
        raise NarrativeExtractorRuntimeError(
            "manifest_output_binding_mismatch"
        )
    return manifest


def _write_bytes_once(path: Path, raw: bytes) -> None:
    if not raw:
        raise NarrativeExtractorRuntimeError("output_empty")
    absolute, parent_descriptor = _open_trusted_directory(
        _absolute_path(path).parent,
        final_mode=0o700,
        final_owner_current=True,
    )
    del absolute
    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                _absolute_path(path).name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise NarrativeExtractorRuntimeError(
                "output_target_not_fresh"
            ) from exc
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise NarrativeExtractorRuntimeError(
                    "output_write_failed"
                )
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(raw)
        ):
            raise NarrativeExtractorRuntimeError(
                "output_file_metadata_invalid"
            )
        os.close(descriptor)
        descriptor = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def write_private_output_once(
    path: Path,
    *,
    pack: StoryOnlyInputPack,
    execution_closure: ExecutionClosure,
    results: Sequence[Mapping[str, object]],
) -> None:
    _write_bytes_once(
        path,
        encode_private_output(
            pack=pack,
            execution_closure=execution_closure,
            results=results,
        ),
    )
