"""One-shot public, non-scoring LoCoMo source acquisition and qualification.

The only semantic access performed by this module is one strict parse of the
pinned public ``data/locomo10.json`` after all three official files have passed
their frozen byte-size, Git-blob, and (where available) SHA-256 bindings.
Receipts contain aggregate topology/capacity facts only.  They never contain a
conversation/sample identifier, speaker, date, turn, question, answer, dia_id,
evidence value, cohort assignment, secret, action, qrel, evaluator, or score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, BinaryIO
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


VERSION = "locomo_p0_public_source_qualification_v1"
STUDY_ID = "LOCOMO_P0_PUBLIC_SCHEMA_TOPOLOGY_V1"
OFFICIAL_REPOSITORY = "https://github.com/snap-research/locomo"
OFFICIAL_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
OFFICIAL_TREE = "cab0c7a94159ac541050229f59c28c08ce7d56a9"

CATEGORY_NAMES = {
    1: "MULTI_HOP",
    2: "TEMPORAL",
    3: "OPEN_DOMAIN",
    4: "SINGLE_HOP",
    5: "ADVERSARIAL",
}
P1_FAMILY_CATEGORY_IDS = (1, 2, 4)
PER_CONVERSATION_PER_FAMILY_QUOTA = 12
PARTITION_CONVERSATION_COUNTS = {
    "A_form_and_label_free_F_search": 2,
    "A_hold": 4,
    "M_search": 4,
}
EXPECTED_CONVERSATION_COUNT = 10
MIN_QREL_CARDINALITY = 1
MAX_QREL_CARDINALITY = 5

MAX_SOURCE_BYTES = 4_000_000
MAX_SESSION_COUNT = 64
MAX_TURNS_PER_SESSION = 10_000
MAX_QA_PER_CONVERSATION = 10_000
READ_CHUNK_BYTES = 1 << 20
HTTP_TIMEOUT_SECONDS = 300

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SESSION = re.compile(r"session_([1-9][0-9]*)\Z")
_SESSION_DATE = re.compile(r"session_([1-9][0-9]*)_date_time\Z")
_DIA_ID = re.compile(r"D([1-9][0-9]*):([1-9][0-9]*)\Z")

SAMPLE_KEYS = frozenset(
    {
        "sample_id",
        "conversation",
        "observation",
        "session_summary",
        "event_summary",
        "qa",
    }
)
TURN_REQUIRED_KEYS = frozenset({"speaker", "dia_id", "text"})
TURN_OPTIONAL_KEYS = frozenset({"img_url", "blip_caption", "query"})
QA_ALLOWED_KEYS = frozenset(
    {"question", "answer", "category", "evidence", "adversarial_answer"}
)


class LocomoP0QualificationError(RuntimeError):
    """The frozen source or one-shot qualification contract failed closed."""


@dataclass(frozen=True)
class SourceFileContract:
    key: str
    relative_path: str
    size_bytes: int
    git_blob_sha1: str
    raw_url: str
    file_sha256: str | None
    semantic_json: bool

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        parsed = urlsplit(self.raw_url)
        if (
            not self.key
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or type(self.size_bytes) is not int
            or self.size_bytes < 1
            or _HEX40.fullmatch(self.git_blob_sha1) is None
            or (
                self.file_sha256 is not None
                and _HEX64.fullmatch(self.file_sha256) is None
            )
            or parsed.scheme != "https"
            or parsed.hostname != "raw.githubusercontent.com"
            or type(self.semantic_json) is not bool
        ):
            raise LocomoP0QualificationError("source file contract is invalid")


OFFICIAL_FILES = {
    "license": SourceFileContract(
        "license",
        "LICENSE.txt",
        19_347,
        "fe463e0f7888bbf8b82e42d55f5743508ddafb7e",
        "https://raw.githubusercontent.com/snap-research/locomo/"
        + OFFICIAL_COMMIT
        + "/LICENSE.txt",
        "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0",
        False,
    ),
    "readme": SourceFileContract(
        "readme",
        "README.MD",
        7_109,
        "8418f7637a75434f76ea1531e0c340996962c488",
        "https://raw.githubusercontent.com/snap-research/locomo/"
        + OFFICIAL_COMMIT
        + "/README.MD",
        "9f8e6fd00a3400aa687109f40ed53715f0a2c028ee3f8c465bdfa96475640e8a",
        False,
    ),
    "data": SourceFileContract(
        "data",
        "data/locomo10.json",
        2_805_274,
        "d95b872480b413d935821fdc3c84f8a8f5f29e73",
        "https://raw.githubusercontent.com/snap-research/locomo/"
        + OFFICIAL_COMMIT
        + "/data/locomo10.json",
        None,
        True,
    ),
}


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LocomoP0QualificationError("value is not canonical JSON") from exc
    return rendered + (b"\n" if newline else b"")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise LocomoP0QualificationError("body already contains self hash")
    result = dict(body)
    result["self_sha256"] = _stable_hash(result)
    return result


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()  # noqa: S324 - Git identity.


def _safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(part in {"", ".", ".."} for part in parts.parts):
        raise LocomoP0QualificationError("source relative path is unsafe")
    destination = root.joinpath(*parts.parts)
    if root != destination and root not in destination.parents:
        raise LocomoP0QualificationError("source path escaped root")
    return destination


def _read_bound_file(path: Path, contract: SourceFileContract) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise LocomoP0QualificationError("pinned source file is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != contract.size_bytes
    ):
        raise LocomoP0QualificationError("pinned source file identity drifted")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                raise LocomoP0QualificationError("source changed during open")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, READ_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise LocomoP0QualificationError("pinned source file read failed") from exc
    raw = b"".join(chunks)
    if (
        len(raw) != contract.size_bytes
        or _git_blob_sha1(raw) != contract.git_blob_sha1
        or (
            contract.file_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != contract.file_sha256
        )
    ):
        raise LocomoP0QualificationError("pinned source byte identity drifted")
    return raw


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise LocomoP0QualificationError(
                "source JSON contains a duplicate or non-string object key"
            )
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise LocomoP0QualificationError("source JSON contains a non-finite number")


def _strict_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LocomoP0QualificationError("source is not strict UTF-8") from exc
    if text.startswith("\ufeff"):
        raise LocomoP0QualificationError("source has a forbidden UTF-8 BOM")
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise LocomoP0QualificationError("source is not strict JSON") from exc


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unsupported"


def _unknown_key_token(key: object) -> str:
    if not isinstance(key, str):
        return "NON_STRING_KEY"
    return "UNKNOWN_KEY_SHA256_" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_answer(value: object) -> bool:
    if _nonempty_text(value):
        return True
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_nonempty_text(member) for member in value)
    )


def _normalize_evidence_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
    return normalized if _DIA_ID.fullmatch(normalized) else None


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _observe_source(root: object) -> dict[str, Any]:
    anomalies: Counter[str] = Counter()
    field_types: Counter[str] = Counter()
    sample_key_contracts: Counter[str] = Counter()
    turn_key_contracts: Counter[str] = Counter()
    qa_key_contracts: Counter[str] = Counter()
    category_counts: Counter[int] = Counter()
    eligible_counts: Counter[int] = Counter()
    evidence_cardinality: Counter[int] = Counter()
    session_count_histogram: Counter[int] = Counter()
    turn_count_histogram: Counter[int] = Counter()
    per_conversation_capacity: list[Counter[int]] = []
    sample_ids: set[str] = set()
    total_turns = 0
    total_qas = 0
    total_evidence_links = 0
    matched_evidence_links = 0

    if not isinstance(root, list):
        anomalies["root_not_array"] += 1
        rows: list[object] = []
    else:
        rows = root

    for sample in rows:
        capacity: Counter[int] = Counter()
        per_conversation_capacity.append(capacity)
        if not isinstance(sample, dict):
            anomalies["sample_not_object"] += 1
            continue
        unknown_sample = set(sample) - SAMPLE_KEYS
        missing_sample = SAMPLE_KEYS - set(sample)
        if not unknown_sample and not missing_sample:
            sample_key_contracts["EXACT"] += 1
        for key in sorted(unknown_sample):
            anomalies["sample_" + _unknown_key_token(key)] += 1
        for key in sorted(missing_sample):
            anomalies["sample_missing_" + key] += 1
        for key in SAMPLE_KEYS:
            field_types[f"sample.{key}:{_json_type(sample.get(key))}"] += 1

        sample_id = sample.get("sample_id")
        if not _nonempty_text(sample_id):
            anomalies["sample_id_not_nonempty_string"] += 1
        elif sample_id in sample_ids:
            anomalies["sample_id_duplicate"] += 1
        else:
            sample_ids.add(sample_id)

        for key in ("observation", "session_summary", "event_summary"):
            if not isinstance(sample.get(key), dict):
                anomalies[f"{key}_not_object"] += 1

        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            anomalies["conversation_not_object"] += 1
            continue
        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")
        if not _nonempty_text(speaker_a):
            anomalies["speaker_a_not_nonempty_string"] += 1
        if not _nonempty_text(speaker_b):
            anomalies["speaker_b_not_nonempty_string"] += 1
        if _nonempty_text(speaker_a) and speaker_a == speaker_b:
            anomalies["speaker_names_not_distinct"] += 1
        speakers = {speaker_a, speaker_b}

        sessions: dict[int, object] = {}
        dates: dict[int, object] = {}
        for key, value in conversation.items():
            if key in {"speaker_a", "speaker_b"}:
                continue
            session_match = _SESSION.fullmatch(key)
            date_match = _SESSION_DATE.fullmatch(key)
            if session_match is not None:
                sessions[int(session_match.group(1))] = value
            elif date_match is not None:
                dates[int(date_match.group(1))] = value
            else:
                anomalies["conversation_" + _unknown_key_token(key)] += 1
        session_numbers = sorted(sessions)
        if (
            not session_numbers
            or session_numbers != list(range(1, len(session_numbers) + 1))
            or len(session_numbers) > MAX_SESSION_COUNT
        ):
            anomalies["session_number_contract_failed"] += 1
        if set(sessions) != set(dates):
            anomalies["session_date_pair_contract_failed"] += 1
        session_count_histogram[len(session_numbers)] += 1

        dia_ids: set[str] = set()
        conversation_turn_count = 0
        for session_number in session_numbers:
            date_value = dates.get(session_number)
            if not _nonempty_text(date_value):
                anomalies["session_date_not_nonempty_string"] += 1
            turns = sessions[session_number]
            if (
                not isinstance(turns, list)
                or not turns
                or len(turns) > MAX_TURNS_PER_SESSION
            ):
                anomalies["session_turn_array_contract_failed"] += 1
                continue
            for turn in turns:
                total_turns += 1
                conversation_turn_count += 1
                if not isinstance(turn, dict):
                    anomalies["turn_not_object"] += 1
                    continue
                unknown = set(turn) - TURN_REQUIRED_KEYS - TURN_OPTIONAL_KEYS
                missing = TURN_REQUIRED_KEYS - set(turn)
                if not unknown and not missing:
                    turn_key_contracts[
                        "EXACT_TEXT_ONLY"
                        if not (set(turn) & TURN_OPTIONAL_KEYS)
                        else "EXACT_MULTIMODAL"
                    ] += 1
                for key in sorted(unknown):
                    anomalies["turn_" + _unknown_key_token(key)] += 1
                for key in sorted(missing):
                    anomalies["turn_missing_" + key] += 1
                optional = set(turn) & TURN_OPTIONAL_KEYS
                if optional and optional != TURN_OPTIONAL_KEYS:
                    anomalies["turn_partial_multimodal_bundle"] += 1
                if turn.get("speaker") not in speakers:
                    anomalies["turn_speaker_not_declared"] += 1
                if not _nonempty_text(turn.get("text")):
                    anomalies["turn_text_not_nonempty_string"] += 1
                for key in optional:
                    if not _nonempty_text(turn.get(key)):
                        anomalies["turn_optional_value_not_nonempty_string"] += 1
                dia_id = turn.get("dia_id")
                match = _DIA_ID.fullmatch(dia_id) if isinstance(dia_id, str) else None
                if match is None:
                    anomalies["turn_dia_id_grammar_invalid"] += 1
                elif int(match.group(1)) != session_number:
                    anomalies["turn_dia_id_session_mismatch"] += 1
                elif dia_id in dia_ids:
                    anomalies["turn_dia_id_duplicate"] += 1
                else:
                    dia_ids.add(dia_id)
        turn_count_histogram[conversation_turn_count] += 1

        qas = sample.get("qa")
        if (
            not isinstance(qas, list)
            or not qas
            or len(qas) > MAX_QA_PER_CONVERSATION
        ):
            anomalies["qa_array_contract_failed"] += 1
            continue
        for qa in qas:
            total_qas += 1
            if not isinstance(qa, dict):
                anomalies["qa_not_object"] += 1
                continue
            unknown = set(qa) - QA_ALLOWED_KEYS
            if not unknown:
                qa_key_contracts["KNOWN_KEYS_ONLY"] += 1
            for key in sorted(unknown):
                anomalies["qa_" + _unknown_key_token(key)] += 1
            for key in ("question", "answer", "category", "evidence"):
                field_types[f"qa.{key}:{_json_type(qa.get(key))}"] += 1
            if not _nonempty_text(qa.get("question")):
                anomalies["qa_question_not_nonempty_string"] += 1
            category = qa.get("category")
            if type(category) is not int or category not in CATEGORY_NAMES:
                anomalies["qa_category_not_official_integer"] += 1
                continue
            category_counts[category] += 1
            if category != 5 and not _valid_answer(qa.get("answer")):
                anomalies["qa_answer_invalid_for_category_1_to_4"] += 1
            if category == 5 and not (
                _valid_answer(qa.get("answer"))
                or _valid_answer(qa.get("adversarial_answer"))
            ):
                anomalies["qa_adversarial_answer_missing"] += 1

            evidence = qa.get("evidence")
            if not isinstance(evidence, list):
                anomalies["qa_evidence_not_array"] += 1
                continue
            normalized_evidence: list[str] = []
            row_evidence_valid = True
            for value in evidence:
                total_evidence_links += 1
                normalized = _normalize_evidence_id(value)
                if normalized is None:
                    anomalies["evidence_dia_id_grammar_invalid"] += 1
                    row_evidence_valid = False
                elif normalized not in dia_ids:
                    anomalies["evidence_dia_id_not_in_conversation"] += 1
                    row_evidence_valid = False
                else:
                    matched_evidence_links += 1
                    normalized_evidence.append(normalized)
            if len(normalized_evidence) != len(set(normalized_evidence)):
                anomalies["evidence_dia_id_duplicate_within_qa"] += 1
                row_evidence_valid = False
            cardinality = len(set(normalized_evidence))
            evidence_cardinality[cardinality] += 1
            eligible = (
                row_evidence_valid
                and MIN_QREL_CARDINALITY <= cardinality <= MAX_QREL_CARDINALITY
                and category in P1_FAMILY_CATEGORY_IDS
                and _nonempty_text(qa.get("question"))
                and _valid_answer(qa.get("answer"))
            )
            if eligible:
                eligible_counts[category] += 1
                capacity[category] += 1

    capacity_summary: dict[str, Any] = {}
    every_conversation_meets_quota = len(per_conversation_capacity) == len(rows)
    for category in P1_FAMILY_CATEGORY_IDS:
        values = [capacity[category] for capacity in per_conversation_capacity]
        distribution = Counter(values)
        category_name = CATEGORY_NAMES[category]
        capacity_summary[category_name] = {
            "category_id": category,
            "maximum_per_conversation": max(values, default=0),
            "minimum_per_conversation": min(values, default=0),
            "per_conversation_count_histogram": _counter(distribution),
            "quota": PER_CONVERSATION_PER_FAMILY_QUOTA,
            "quota_satisfying_conversation_count": sum(
                value >= PER_CONVERSATION_PER_FAMILY_QUOTA for value in values
            ),
            "total_eligible_count": sum(values),
        }
        every_conversation_meets_quota &= (
            len(values) == EXPECTED_CONVERSATION_COUNT
            and all(
                value >= PER_CONVERSATION_PER_FAMILY_QUOTA for value in values
            )
        )

    conversation_count_ok = len(rows) == EXPECTED_CONVERSATION_COUNT
    partition_shape_ok = (
        sum(PARTITION_CONVERSATION_COUNTS.values())
        == EXPECTED_CONVERSATION_COUNT
    )
    total_anomalies = sum(anomalies.values())
    feasible = (
        conversation_count_ok
        and partition_shape_ok
        and every_conversation_meets_quota
        and total_anomalies == 0
    )
    return {
        "aggregate_counts": {
            "conversation_count": len(rows),
            "matched_evidence_link_count": matched_evidence_links,
            "qa_count": total_qas,
            "sample_id_duplicate_count": anomalies["sample_id_duplicate"],
            "turn_count": total_turns,
            "evidence_link_count": total_evidence_links,
        },
        "category_count": {
            CATEGORY_NAMES[key]: category_counts[key]
            for key in sorted(CATEGORY_NAMES)
        },
        "eligible_evidence_bearing_count": {
            CATEGORY_NAMES[key]: eligible_counts[key]
            for key in P1_FAMILY_CATEGORY_IDS
        },
        "evidence_cardinality_histogram": _counter(evidence_cardinality),
        "family_capacity": capacity_summary,
        "field_type_count": _counter(field_types),
        "partition_feasibility": {
            "all_conversations_meet_every_family_quota": (
                every_conversation_meets_quota
            ),
            "conversation_count_exactly_ten": conversation_count_ok,
            "fixed_partition_shape": dict(PARTITION_CONVERSATION_COUNTS),
            "partition_feasible_without_selecting_conversations": feasible,
            "selected_conversation_count": 0,
        },
        "qa_key_contract_count": _counter(qa_key_contracts),
        "sample_key_contract_count": _counter(sample_key_contracts),
        "schema_anomaly_count": _counter(anomalies),
        "session_count_histogram": _counter(session_count_histogram),
        "total_schema_anomaly_count": total_anomalies,
        "turn_count_histogram": _counter(turn_count_histogram),
        "turn_key_contract_count": _counter(turn_key_contracts),
    }


def qualify_source(
    *,
    source_root: Path,
    expected_files: Mapping[str, SourceFileContract] = OFFICIAL_FILES,
) -> dict[str, Any]:
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise LocomoP0QualificationError("source root is not a directory")
    if set(expected_files) != {"data", "license", "readme"}:
        raise LocomoP0QualificationError("source registry is incomplete")

    raw_files: dict[str, bytes] = {}
    source_receipts: dict[str, Any] = {}
    for key in ("license", "readme", "data"):
        contract = expected_files[key]
        raw = _read_bound_file(_safe_path(root, contract.relative_path), contract)
        raw_files[key] = raw
        source_receipts[key] = {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": contract.git_blob_sha1,
            "relative_path": contract.relative_path,
            "size_bytes": len(raw),
        }

    observation = _observe_source(_strict_json(raw_files["data"]))
    qualified = observation["partition_feasibility"][
        "partition_feasible_without_selecting_conversations"
    ]
    body = {
        "access_boundary": {
            "action_evaluator_qrel_or_score_count": 0,
            "conversation_cohort_or_secret_count": 0,
            "individual_source_value_output_count": 0,
            "public_data_JSON_decode_count": 1,
            "source_file_identity_read_count": 3,
        },
        "category_mapping": {
            str(key): CATEGORY_NAMES[key] for key in sorted(CATEGORY_NAMES)
        },
        "official_commit": OFFICIAL_COMMIT,
        "official_tree": OFFICIAL_TREE,
        "qualification": observation,
        "recorded_date": "2026-07-27",
        "schema": VERSION,
        "source_files": source_receipts,
        "status": (
            "qualified_public_non_scoring_schema_topology"
            if qualified
            else "terminal_not_qualified_no_same_source_revision"
        ),
        "study_id": STUDY_ID,
    }
    return _self_hashed(body)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = _canonical_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if stat.S_IMODE(destination.stat().st_mode) != 0o600:
            raise LocomoP0QualificationError("output mode is not 0600")
    except OSError as exc:
        raise LocomoP0QualificationError("output was not written exclusively") from exc


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _default_open(url: str) -> BinaryIO:
    request = Request(
        url,
        method="GET",
        headers={"Accept": "application/octet-stream", "User-Agent": VERSION},
    )
    opener = build_opener(ProxyHandler({}), _RejectRedirects())
    return opener.open(request, timeout=HTTP_TIMEOUT_SECONDS)


def _download_bound_file(
    *,
    contract: SourceFileContract,
    destination: Path,
    opener: Callable[[str], BinaryIO],
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    size = 0
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(  # noqa: S324 - Git identity.
        f"blob {contract.size_bytes}\0".encode("ascii")
    )
    try:
        response = opener(contract.raw_url)
        with response:
            final_url = getattr(response, "geturl", lambda: contract.raw_url)()
            if final_url != contract.raw_url:
                raise LocomoP0QualificationError("source redirect is forbidden")
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                while True:
                    chunk = response.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise LocomoP0QualificationError(
                            "download returned non-byte content"
                        )
                    size += len(chunk)
                    if size > min(MAX_SOURCE_BYTES, contract.size_bytes):
                        raise LocomoP0QualificationError(
                            "download exceeded frozen size"
                        )
                    sha256.update(chunk)
                    git_sha1.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    if (
        size != contract.size_bytes
        or git_sha1.hexdigest() != contract.git_blob_sha1
        or (
            contract.file_sha256 is not None
            and sha256.hexdigest() != contract.file_sha256
        )
    ):
        raise LocomoP0QualificationError("downloaded source identity mismatch")
    return {
        "file_sha256": sha256.hexdigest(),
        "git_blob_sha1": git_sha1.hexdigest(),
        "size_bytes": size,
    }


def _verify_frozen_manifests(project_root: Path) -> None:
    required = {
        "manifests/locomo_p0_public_source_custody_v1.json": (
            "locomo_p0_public_source_custody_v1"
        ),
        "manifests/locomo_p0_public_schema_qualification_freeze_v1.json": (
            "locomo_p0_public_schema_qualification_freeze_v1"
        ),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for relative, schema in required.items():
        path = project_root / relative
        try:
            value = _strict_json(path.read_bytes())
        except OSError as exc:
            raise LocomoP0QualificationError("frozen manifest unavailable") from exc
        if not isinstance(value, dict) or value.get("schema") != schema:
            raise LocomoP0QualificationError("frozen manifest schema drifted")
        declared = value.pop("self_sha256", None)
        if not isinstance(declared, str) or declared != _stable_hash(value):
            raise LocomoP0QualificationError("frozen manifest self hash drifted")
        loaded[schema] = value

    custody = loaded["locomo_p0_public_source_custody_v1"]
    official = custody.get("official_source")
    if (
        not isinstance(official, dict)
        or official.get("commit") != OFFICIAL_COMMIT
        or official.get("tree") != OFFICIAL_TREE
    ):
        raise LocomoP0QualificationError("custody source identity drifted")
    frozen_files = official.get("files")
    if not isinstance(frozen_files, dict):
        raise LocomoP0QualificationError("custody file registry drifted")
    for key, contract in OFFICIAL_FILES.items():
        row = frozen_files.get(key)
        if (
            not isinstance(row, dict)
            or row.get("relative_path") != contract.relative_path
            or row.get("size_bytes") != contract.size_bytes
            or row.get("git_blob_sha1") != contract.git_blob_sha1
            or (
                contract.file_sha256 is not None
                and row.get("file_sha256") != contract.file_sha256
            )
        ):
            raise LocomoP0QualificationError("custody file binding drifted")

    freeze = loaded["locomo_p0_public_schema_qualification_freeze_v1"]
    family = freeze.get("family_and_capacity_contract")
    if (
        not isinstance(family, dict)
        or family.get("fixed_P1_family_category_ids")
        != list(P1_FAMILY_CATEGORY_IDS)
        or family.get("per_conversation_per_family_quota")
        != PER_CONVERSATION_PER_FAMILY_QUOTA
        or family.get("required_conversation_count")
        != EXPECTED_CONVERSATION_COUNT
        or family.get("fixed_conversation_group_partition_shape")
        != PARTITION_CONVERSATION_COUNTS
        or family.get("minimum_and_maximum_qrel_cardinality")
        != [MIN_QREL_CARDINALITY, MAX_QREL_CARDINALITY]
    ):
        raise LocomoP0QualificationError("qualification capacity freeze drifted")
    implementation = freeze.get("implementation_binding")
    if not isinstance(implementation, dict):
        raise LocomoP0QualificationError("implementation binding is missing")
    for row in implementation.values():
        if not isinstance(row, dict):
            raise LocomoP0QualificationError("implementation binding drifted")
        relative = row.get("relative_path")
        expected_sha256 = row.get("file_sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha256, str)
            or _HEX64.fullmatch(expected_sha256) is None
        ):
            raise LocomoP0QualificationError("implementation binding drifted")
        path = _safe_path(project_root, relative)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise LocomoP0QualificationError(
                "bound implementation file is unavailable"
            ) from exc
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise LocomoP0QualificationError(
                "bound implementation file hash drifted"
            )


def acquire_and_qualify(
    *,
    project_root: Path,
    work_root: Path,
    opener: Callable[[str], BinaryIO] = _default_open,
    expected_files: Mapping[str, SourceFileContract] = OFFICIAL_FILES,
    manifest_verifier: Callable[[Path], None] = _verify_frozen_manifests,
) -> dict[str, Any]:
    project = project_root.resolve(strict=True)
    manifest_verifier(project)
    work = work_root.resolve()
    if work.exists() or work.is_symlink():
        raise LocomoP0QualificationError("one-shot work root is already consumed")
    work.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    work.mkdir(mode=0o700)
    write_json_exclusive(
        work / "qualification.attempt.private.json",
        _self_hashed(
            {
                "data_body_decode_count": 0,
                "official_commit": OFFICIAL_COMMIT,
                "schema": "locomo_p0_qualification_attempt_v1",
                "secret_action_qrel_or_score_count": 0,
                "status": "attempt_claimed_before_network",
                "study_id": STUDY_ID,
            }
        ),
    )
    source_root = work / "source"
    source_root.mkdir(mode=0o700)
    completed = 0
    try:
        for key in ("license", "readme", "data"):
            contract = expected_files[key]
            _download_bound_file(
                contract=contract,
                destination=_safe_path(source_root, contract.relative_path),
                opener=opener,
            )
            completed += 1
        receipt = qualify_source(
            source_root=source_root,
            expected_files=expected_files,
        )
        write_json_exclusive(work / "qualification.receipt.safe.json", receipt)
        if not receipt["status"].startswith("qualified_"):
            raise LocomoP0QualificationError(
                "source failed frozen non-scoring qualification"
            )
        return receipt
    except BaseException as exc:
        try:
            write_json_exclusive(
                work / "qualification.terminal_failure.safe.json",
                _self_hashed(
                    {
                        "completed_source_file_count": completed,
                        "failure_class": type(exc).__name__,
                        "individual_source_value_output_count": 0,
                        "retry_or_contract_revision_authorized": False,
                        "schema": "locomo_p0_qualification_terminal_failure_v1",
                        "status": "terminal_no_retry",
                        "study_id": STUDY_ID,
                    }
                ),
            )
        except BaseException:
            pass
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("qualify-local")
    local.add_argument("--source-root", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    acquire = subparsers.add_parser("acquire-and-qualify")
    acquire.add_argument("--project-root", type=Path, required=True)
    acquire.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "qualify-local":
        receipt = qualify_source(source_root=args.source_root)
        write_json_exclusive(args.output, receipt)
    else:
        receipt = acquire_and_qualify(
            project_root=args.project_root,
            work_root=args.work_root,
        )
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
