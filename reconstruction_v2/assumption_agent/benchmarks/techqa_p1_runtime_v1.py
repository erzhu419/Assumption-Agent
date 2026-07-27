"""One-shot offline production runtime for the frozen TechQA P1 study.

This module is deliberately an orchestration boundary, not another policy or
effect gate.  It binds an already-qualified three-file TechQA projection,
selects the frozen cohorts with one pre-existing HMAC secret, streams only the
selected candidate documents into memory, delegates all RAW/E0/E1 formation
and scoring to :mod:`techqa_p1_formal_v1`, and schedules the already-qualified
official HippoRAG adapter for A_hold in two fixed two-GPU waves.  M_search is
the evaluator-only E1-versus-E0 L5 block and never invokes HippoRAG.

The outer systemd unit is responsible for denying network access.  The
production launcher additionally supplies only an allow-listed offline child
environment and audits IP-family syscalls with the configured ``strace``
binary.  There is no API or online evaluator path.

Every formal root is claimed durably before a benchmark source file is
opened.  A failure consumes the attempt and produces a safe aggregate
terminal plus a private failure record; it never authorizes retry, replay,
resampling, a provider/model/candidate change, or a new gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import threading
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import averitec_p1_runtime_v1 as base_runtime
from assumption_agent.benchmarks import (
    techqa_p0_public_source_qualification_v1 as p0,
)
from assumption_agent.benchmarks import techqa_p1_formal_v1 as formal
from assumption_agent.benchmarks import (
    techqa_p1_official_hipporag_v1 as hippo_adapter,
)


VERSION = "techqa_p1_runtime_v1"
CONFIG_SCHEMA = f"{VERSION}_formal_config_v1"
ATTEMPT_SCHEMA = f"{VERSION}_formal_attempt_v1"
PRIVATE_ARCHIVE_SCHEMA = f"{VERSION}_private_archive_v1"
SAFE_TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
FAILURE_TERMINAL_SCHEMA = f"{VERSION}_safe_failure_terminal_v1"
LAUNCH_RECEIPT_SCHEMA = f"{VERSION}_private_hipporag_launch_v1"

CONFIG_MODE = 0o400
WORK_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
FROZEN_FILE_MODE = 0o400
SAFE_TERMINAL_MODE = 0o444
ALLOWED_ASSET_MODES = frozenset({0o400, 0o440, 0o444, 0o600})

GPU_IDS = ("0", "1")
HIPPO_CLUSTERS_PER_STAGE = 4
HIPPO_MAX_CONCURRENCY = 2
HIPPO_WAVES = ((0, 1), (2, 3))
MAX_QA_BYTES = p0.MAX_QUERY_MEMBER_BYTES
MAX_CORPUS_BYTES = p0.MAX_CORPUS_MEMBER_BYTES
MAX_SELECTED_QUESTION_COUNT = sum(
    formal.BLOCK_FAMILY_QUOTAS[block] * len(formal.FAMILIES)
    for block in formal.BLOCKS
)
MAX_SELECTED_CANDIDATE_REFERENCES = (
    MAX_SELECTED_QUESTION_COUNT * formal.CANDIDATE_DOCUMENT_COUNT
)
MAX_SELECTED_DOCUMENT_COUNT = MAX_SELECTED_CANDIDATE_REFERENCES
DEFAULT_HIPPO_TIMEOUT_SECONDS = 14_400
MAX_HIPPO_TIMEOUT_SECONDS = 86_400
MAX_WORKER_LOG_BYTES = 128 * 1024 * 1024

P0_RECEIPT_SCHEMA = f"{p0.VERSION}_safe_aggregate_receipt"
P0_QUALIFIED_STATUS = (
    "qualified_public_non_scoring_schema_and_family_capacity"
)
P0_ELIGIBILITY_SCHEMA = (
    f"{p0.VERSION}_private_eligibility_manifest_v1"
)
P0_ELIGIBILITY_RULE_VERSION = p0.ELIGIBILITY_RULE_VERSION
OFFICIAL_ADAPTER_MODULE = (
    "assumption_agent.benchmarks.techqa_p1_official_hipporag_v1"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_STAGE = re.compile(r"[A-Za-z0-9_]+\Z")


class TechqaP1RuntimeError(RuntimeError):
    """A frozen runtime, custody, source, or one-shot invariant failed."""


def canonical_bytes(
    value: object,
    *,
    newline: bool = True,
) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TechqaP1RuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise TechqaP1RuntimeError("self hash already exists")
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def _verify_self(value: Mapping[str, object], *, field: str) -> str:
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or not hmac.compare_digest(stable_hash(body), claimed)
    ):
        raise TechqaP1RuntimeError(f"{field} self hash drifted")
    return claimed


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise TechqaP1RuntimeError(f"{field} is not a SHA-256 digest")
    return value


def _strict_int(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TechqaP1RuntimeError(f"{field} is outside its frozen bound")
    return value


def _direct_absolute(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise TechqaP1RuntimeError(f"{field} is not a direct absolute path")
    return candidate


def _regular_metadata(
    path: Path,
    *,
    field: str,
    mode: int,
    maximum_bytes: int | None = None,
) -> os.stat_result:
    path = _direct_absolute(path, field=field)
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TechqaP1RuntimeError(f"{field} is unavailable") from exc
    if (
        resolved != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != mode
        or info.st_nlink != 1
        or info.st_size < 1
        or (maximum_bytes is not None and info.st_size > maximum_bytes)
    ):
        raise TechqaP1RuntimeError(f"{field} metadata drifted")
    return info


def _open_direct_read(path: Path, *, field: str):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        return os.fdopen(descriptor, "rb")
    except OSError as exc:
        raise TechqaP1RuntimeError(f"{field} cannot be opened") from exc


def _read_regular(
    path: Path,
    *,
    field: str,
    mode: int,
    maximum_bytes: int,
    expected_sha256: str | None = None,
) -> bytes:
    info = _regular_metadata(
        path,
        field=field,
        mode=mode,
        maximum_bytes=maximum_bytes,
    )
    with _open_direct_read(path, field=field) as handle:
        raw = handle.read(maximum_bytes + 1)
    if len(raw) != info.st_size or len(raw) > maximum_bytes:
        raise TechqaP1RuntimeError(f"{field} byte bound drifted")
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(
        actual, expected_sha256
    ):
        raise TechqaP1RuntimeError(f"{field} SHA-256 drifted")
    return raw


def _private_directory(path: Path, *, fresh: bool) -> None:
    path = _direct_absolute(path, field="private directory")
    try:
        path.mkdir(
            parents=True,
            mode=WORK_DIRECTORY_MODE,
            exist_ok=not fresh,
        )
    except OSError as exc:
        raise TechqaP1RuntimeError(
            "private directory cannot be created"
        ) from exc
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TechqaP1RuntimeError(
            "private directory disappeared"
        ) from exc
    if (
        path.is_symlink()
        or resolved != path
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != WORK_DIRECTORY_MODE
    ):
        raise TechqaP1RuntimeError("private directory mode drifted")


def _write_once(path: Path, raw: bytes, *, mode: int) -> str:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise TechqaP1RuntimeError("one-shot output path is not fresh")
    _private_directory(path.parent, fresh=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise TechqaP1RuntimeError(
            "one-shot output cannot be written"
        ) from exc
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != mode
        or path.read_bytes() != raw
    ):
        raise TechqaP1RuntimeError("one-shot output verification failed")
    return hashlib.sha256(raw).hexdigest()


def _write_json_once(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int,
) -> str:
    return _write_once(path, canonical_bytes(value), mode=mode)


def _read_canonical_json(
    path: Path,
    *,
    field: str,
    mode: int,
    expected_sha256: str | None = None,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> dict[str, object]:
    raw = _read_regular(
        path,
        field=field,
        mode=mode,
        maximum_bytes=maximum_bytes,
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TechqaP1RuntimeError(
            f"{field} is not canonical JSON"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise TechqaP1RuntimeError(f"{field} canonical bytes drifted")
    return value


@dataclass(frozen=True, slots=True)
class AssetBinding:
    path: Path
    sha256: str
    mode: int

    @classmethod
    def from_payload(
        cls,
        value: object,
        *,
        field: str,
    ) -> "AssetBinding":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"mode", "path", "sha256"}
        ):
            raise TechqaP1RuntimeError(f"{field} binding drifted")
        mode = value.get("mode")
        if type(mode) is not int or mode not in ALLOWED_ASSET_MODES:
            raise TechqaP1RuntimeError(f"{field} mode binding drifted")
        return cls(
            path=_direct_absolute(str(value.get("path")), field=field),
            sha256=_sha256(value.get("sha256"), field=f"{field} SHA-256"),
            mode=mode,
        )

    def payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "path": str(self.path),
            "sha256": self.sha256,
        }

    def verify_metadata(
        self,
        *,
        field: str,
        maximum_bytes: int | None = None,
    ) -> os.stat_result:
        return _regular_metadata(
            self.path,
            field=field,
            mode=self.mode,
            maximum_bytes=maximum_bytes,
        )


@dataclass(frozen=True, slots=True)
class RuntimeAssets:
    paths: base_runtime.RuntimePaths
    timeout_seconds: int

    @classmethod
    def from_payload(cls, value: object) -> "RuntimeAssets":
        path_fields = {
            "hipporag_source_root",
            "minilm_model_root",
            "official_base_site_root",
            "official_overlay_root",
            "official_python",
            "p16_site_root",
            "project_root",
            "smollm_model_root",
            "strace_path",
            "typed_python",
            "typed_site_root",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != {
                "gpu_ids",
                "official_timeout_seconds",
                "paths",
            }
            or value.get("gpu_ids") != list(GPU_IDS)
            or not isinstance(value.get("paths"), Mapping)
            or set(value["paths"]) != path_fields
        ):
            raise TechqaP1RuntimeError("runtime asset config drifted")
        timeout = _strict_int(
            value.get("official_timeout_seconds"),
            field="official timeout",
            minimum=1,
            maximum=MAX_HIPPO_TIMEOUT_SECONDS,
        )
        paths = base_runtime.RuntimePaths(
            **{
                key: str(value["paths"][key])  # type: ignore[index]
                for key in sorted(path_fields)
            }
        )
        paths.validate()
        return cls(paths=paths, timeout_seconds=timeout)


@dataclass(frozen=True, slots=True)
class FormalConfig:
    config_self_sha256: str
    work_root: Path
    training_q_a: AssetBinding
    dev_q_a: AssetBinding
    training_dev_technotes: AssetBinding
    qualification_receipt: AssetBinding
    eligibility_manifest: AssetBinding
    implementation_freeze: AssetBinding
    hmac_secret: AssetBinding
    runtime: RuntimeAssets

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "FormalConfig":
        if set(value) != {
            "hmac_secret",
            "eligibility_manifest",
            "implementation_freeze",
            "qualification_receipt",
            "runtime",
            "schema",
            "self_sha256",
            "source",
            "study_id",
            "work_root",
        }:
            raise TechqaP1RuntimeError("formal config envelope drifted")
        config_hash = _verify_self(value, field="formal config")
        if (
            value.get("schema") != CONFIG_SCHEMA
            or value.get("study_id") != formal.STUDY_ID
            or not isinstance(value.get("source"), Mapping)
            or set(value["source"])  # type: ignore[arg-type]
            != {
                p0.CORPUS_BASENAME,
                p0.DEV_QA_BASENAME,
                p0.TRAIN_QA_BASENAME,
            }
        ):
            raise TechqaP1RuntimeError("formal config identity drifted")
        source = value["source"]
        assert isinstance(source, Mapping)
        work_root = _direct_absolute(
            str(value.get("work_root")), field="formal work root"
        )
        if work_root.exists() or work_root.is_symlink():
            raise TechqaP1RuntimeError(
                "formal work root is not fresh; replay is forbidden"
            )
        return cls(
            config_self_sha256=config_hash,
            work_root=work_root,
            training_q_a=AssetBinding.from_payload(
                source[p0.TRAIN_QA_BASENAME],
                field=p0.TRAIN_QA_BASENAME,
            ),
            dev_q_a=AssetBinding.from_payload(
                source[p0.DEV_QA_BASENAME],
                field=p0.DEV_QA_BASENAME,
            ),
            training_dev_technotes=AssetBinding.from_payload(
                source[p0.CORPUS_BASENAME],
                field=p0.CORPUS_BASENAME,
            ),
            qualification_receipt=AssetBinding.from_payload(
                value.get("qualification_receipt"),
                field="P0 qualification receipt",
            ),
            eligibility_manifest=AssetBinding.from_payload(
                value.get("eligibility_manifest"),
                field="P0 private eligibility manifest",
            ),
            implementation_freeze=AssetBinding.from_payload(
                value.get("implementation_freeze"),
                field="implementation freeze",
            ),
            hmac_secret=AssetBinding.from_payload(
                value.get("hmac_secret"),
                field="HMAC secret",
            ),
            runtime=RuntimeAssets.from_payload(value.get("runtime")),
        )

    @property
    def source_bindings(self) -> Mapping[str, AssetBinding]:
        return {
            p0.TRAIN_QA_BASENAME: self.training_q_a,
            p0.DEV_QA_BASENAME: self.dev_q_a,
            p0.CORPUS_BASENAME: self.training_dev_technotes,
        }


def load_config(path: Path) -> FormalConfig:
    value = _read_canonical_json(
        path,
        field="formal config",
        mode=CONFIG_MODE,
    )
    return FormalConfig.from_payload(value)


def _verify_supporting_receipts(
    config: FormalConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        any(
            binding.mode != PRIVATE_FILE_MODE
            for binding in config.source_bindings.values()
        )
        or config.qualification_receipt.mode != PRIVATE_FILE_MODE
        or config.eligibility_manifest.mode != PRIVATE_FILE_MODE
        or config.hmac_secret.mode != FROZEN_FILE_MODE
    ):
        raise TechqaP1RuntimeError(
            "P0 source/receipt/eligibility or secret mode contract drifted"
        )
    qualification = _read_canonical_json(
        config.qualification_receipt.path,
        field="P0 qualification receipt",
        mode=config.qualification_receipt.mode,
        expected_sha256=config.qualification_receipt.sha256,
    )
    _verify_self(qualification, field="P0 qualification receipt")
    archive = qualification.get("archive")
    targets = (
        archive.get("target_members")
        if isinstance(archive, Mapping)
        else None
    )
    access = qualification.get("access_boundary")
    persistence = qualification.get("qualified_source_persistence")
    eligibility_binding = qualification.get(
        "private_eligibility_manifest_binding"
    )
    if (
        qualification.get("schema") != P0_RECEIPT_SCHEMA
        or qualification.get("status") != P0_QUALIFIED_STATUS
        or not isinstance(targets, Mapping)
        or set(targets) != set(config.source_bindings)
        or not isinstance(access, Mapping)
        or access.get("action_model_qrel_evaluator_or_score_count") != 0
        or access.get("cohort_assignment_or_secret_count") != 0
        or access.get("online_or_API_evaluation_count") != 0
        or access.get(
            "source_archive_whitelisted_member_extraction_count"
        )
        != 3
        or not isinstance(persistence, Mapping)
        or persistence.get("exact_private_regular_file_count") != 3
        or persistence.get(
            "full_archive_or_nonwhitelisted_member_persistence_count"
        )
        != 0
        or persistence.get("mode") != "0600"
        or persistence.get(
            "member_byte_identity_verified_against_receipt_count"
        )
        != 3
        or not isinstance(eligibility_binding, Mapping)
        or set(eligibility_binding)
        != {
            "eligible_row_count_by_split",
            "file_sha256",
            "self_sha256",
        }
        or eligibility_binding.get("file_sha256")
        != config.eligibility_manifest.sha256
        or not isinstance(
            eligibility_binding.get("eligible_row_count_by_split"),
            Mapping,
        )
        or set(
            eligibility_binding["eligible_row_count_by_split"]  # type: ignore[index]
        )
        != {"DEV", "TRAIN"}
        or any(
            type(
                eligibility_binding["eligible_row_count_by_split"][split]  # type: ignore[index]
            )
            is not int
            or eligibility_binding[
                "eligible_row_count_by_split"
            ][split]  # type: ignore[index]
            < formal.SOURCE_MINIMUM_FAMILY_COUNTS[split]
            * len(formal.FAMILIES)
            for split in ("TRAIN", "DEV")
        )
        or _HEX64.fullmatch(
            str(eligibility_binding.get("self_sha256"))
        )
        is None
    ):
        raise TechqaP1RuntimeError(
            "P0 qualification receipt is not an eligible zero-score receipt"
        )
    config.eligibility_manifest.verify_metadata(
        field="P0 private eligibility manifest",
        maximum_bytes=64 * 1024 * 1024,
    )
    for basename, binding in config.source_bindings.items():
        row = targets.get(basename)
        if (
            not isinstance(row, Mapping)
            or row.get("content_sha256") != binding.sha256
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 1
        ):
            raise TechqaP1RuntimeError(
                "extracted source does not match the P0 member receipt"
            )
        metadata = binding.verify_metadata(
            field=basename,
            maximum_bytes=(
                MAX_CORPUS_BYTES
                if basename == p0.CORPUS_BASENAME
                else MAX_QA_BYTES
            ),
        )
        if metadata.st_size != row["size_bytes"]:
            raise TechqaP1RuntimeError(
                "extracted source size does not match the P0 receipt"
            )

    implementation = _read_canonical_json(
        config.implementation_freeze.path,
        field="implementation freeze",
        mode=config.implementation_freeze.mode,
        expected_sha256=config.implementation_freeze.sha256,
    )
    _verify_self(implementation, field="implementation freeze")
    if implementation.get("study_id") != formal.STUDY_ID:
        raise TechqaP1RuntimeError(
            "implementation freeze study identity drifted"
        )
    return qualification, implementation


def _attempt_payload(config: FormalConfig) -> dict[str, object]:
    return self_hashed(
        {
            "attempt_count": 1,
            "config_self_sha256": config.config_self_sha256,
            "eligibility_manifest_sha256": (
                config.eligibility_manifest.sha256
            ),
            "hmac_secret_sha256": config.hmac_secret.sha256,
            "implementation_freeze_sha256": (
                config.implementation_freeze.sha256
            ),
            "online_or_API_evaluator_call_count": 0,
            "qualification_receipt_sha256": (
                config.qualification_receipt.sha256
            ),
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": ATTEMPT_SCHEMA,
            "source_sha256": {
                basename: binding.sha256
                for basename, binding in sorted(
                    config.source_bindings.items()
                )
            },
            "study_id": formal.STUDY_ID,
        }
    )


def claim_formal_attempt(config: FormalConfig) -> str:
    """Claim the formal root before opening the secret or benchmark source."""

    _private_directory(config.work_root, fresh=True)
    return _write_json_once(
        config.work_root / "formal.attempt.json",
        _attempt_payload(config),
        mode=FROZEN_FILE_MODE,
    )


def _load_secret(binding: AssetBinding) -> bytes:
    raw = _read_regular(
        binding.path,
        field="HMAC secret",
        mode=binding.mode,
        maximum_bytes=formal.HMAC_SECRET_BYTES,
        expected_sha256=binding.sha256,
    )
    if len(raw) != formal.HMAC_SECRET_BYTES:
        raise TechqaP1RuntimeError(
            "HMAC secret is not exactly the frozen 32 bytes"
        )
    return raw


@dataclass(frozen=True, slots=True)
class EligibleQuestionIdentity:
    question_id: str
    family: str
    normalized_query_sha256: str


@dataclass(frozen=True, slots=True)
class EligibilityManifest:
    rows_by_split: Mapping[
        str, Mapping[str, EligibleQuestionIdentity]
    ]
    self_sha256: str
    file_sha256: str


def _load_eligibility_manifest(
    config: FormalConfig,
    *,
    qualification_receipt: Mapping[str, object],
) -> EligibilityManifest:
    value = _read_canonical_json(
        config.eligibility_manifest.path,
        field="P0 private eligibility manifest",
        mode=config.eligibility_manifest.mode,
        expected_sha256=config.eligibility_manifest.sha256,
        maximum_bytes=64 * 1024 * 1024,
    )
    expected_keys = {
        "cohort_HMAC_action_qrel_evaluator_or_score_count",
        "eligibility_rule_version",
        "eligible_answerable_rows_by_split",
        "eligible_row_count_by_split",
        "schema",
        "self_sha256",
        "source_member_content_sha256",
        "study_id",
    }
    self_sha = _verify_self(
        value, field="P0 private eligibility manifest"
    )
    safe_binding = qualification_receipt.get(
        "private_eligibility_manifest_binding"
    )
    source_hashes = value.get("source_member_content_sha256")
    counts = value.get("eligible_row_count_by_split")
    raw_splits = value.get("eligible_answerable_rows_by_split")
    if (
        set(value) != expected_keys
        or value.get("schema") != P0_ELIGIBILITY_SCHEMA
        or value.get("study_id") != p0.STUDY_ID
        or value.get("eligibility_rule_version")
        != P0_ELIGIBILITY_RULE_VERSION
        or value.get(
            "cohort_HMAC_action_qrel_evaluator_or_score_count"
        )
        != 0
        or not isinstance(source_hashes, Mapping)
        or source_hashes
        != {
            basename: binding.sha256
            for basename, binding in sorted(
                config.source_bindings.items()
            )
        }
        or not isinstance(counts, Mapping)
        or set(counts) != {"DEV", "TRAIN"}
        or not isinstance(raw_splits, Mapping)
        or set(raw_splits) != {"DEV", "TRAIN"}
        or not isinstance(safe_binding, Mapping)
        or safe_binding.get("file_sha256")
        != config.eligibility_manifest.sha256
        or safe_binding.get("self_sha256") != self_sha
        or safe_binding.get("eligible_row_count_by_split") != counts
    ):
        raise TechqaP1RuntimeError(
            "P0 private eligibility manifest binding drifted"
        )
    parsed: dict[str, dict[str, EligibleQuestionIdentity]] = {}
    all_ids: set[str] = set()
    for split in ("TRAIN", "DEV"):
        raw_rows = raw_splits[split]
        if isinstance(raw_rows, (str, bytes)) or not isinstance(
            raw_rows, Sequence
        ):
            raise TechqaP1RuntimeError(
                "P0 eligibility split rows drifted"
            )
        rows: dict[str, EligibleQuestionIdentity] = {}
        for raw_row in raw_rows:
            if (
                not isinstance(raw_row, Mapping)
                or set(raw_row)
                != {
                    "family",
                    "normalized_query_sha256",
                    "question_id",
                }
            ):
                raise TechqaP1RuntimeError(
                    "P0 eligibility row schema drifted"
                )
            try:
                question_id = p0._identifier(
                    raw_row.get("question_id"),
                    field="eligible question ID",
                )
            except p0.TechqaP0QualificationError as exc:
                raise TechqaP1RuntimeError(
                    "P0 eligibility question ID drifted"
                ) from exc
            family = raw_row.get("family")
            normalized = raw_row.get("normalized_query_sha256")
            if (
                family not in formal.FAMILIES
                or not isinstance(normalized, str)
                or _HEX64.fullmatch(normalized) is None
                or question_id in rows
                or question_id in all_ids
            ):
                raise TechqaP1RuntimeError(
                    "P0 eligibility identity registry drifted"
                )
            rows[question_id] = EligibleQuestionIdentity(
                question_id=question_id,
                family=str(family),
                normalized_query_sha256=normalized,
            )
            all_ids.add(question_id)
        if type(counts[split]) is not int or counts[split] != len(rows):
            raise TechqaP1RuntimeError(
                "P0 eligibility row count drifted"
            )
        parsed[split] = rows
    return EligibilityManifest(
        rows_by_split=parsed,
        self_sha256=self_sha,
        file_sha256=config.eligibility_manifest.sha256,
    )


def _parse_answerable_questions(
    *,
    binding: AssetBinding,
    split: str,
    eligible: Mapping[str, EligibleQuestionIdentity],
) -> tuple[formal.VerifiedQuestion, ...]:
    raw = _read_regular(
        binding.path,
        field=f"{split} QA",
        mode=binding.mode,
        maximum_bytes=MAX_QA_BYTES,
        expected_sha256=binding.sha256,
    )
    try:
        rows = p0._strict_json_bytes(raw)
    except p0.TechqaP0QualificationError as exc:
        raise TechqaP1RuntimeError(
            f"{split} QA strict JSON drifted after P0"
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise TechqaP1RuntimeError(f"{split} QA root drifted")
    answerable: list[formal.VerifiedQuestion] = []
    seen_eligible: set[str] = set()
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise TechqaP1RuntimeError(f"{split} QA row drifted")
        try:
            question_id = p0._identifier(
                raw_row.get("QUESTION_ID"), field="QUESTION_ID"
            )
        except p0.TechqaP0QualificationError as exc:
            raise TechqaP1RuntimeError(
                f"{split} QUESTION_ID drifted after P0"
            ) from exc
        identity = eligible.get(question_id)
        if identity is None:
            continue
        if (
            raw_row.get("ANSWERABLE") != "Y"
            or question_id in seen_eligible
        ):
            raise TechqaP1RuntimeError(
                f"{split} eligible answerable coverage drifted"
            )
        try:
            title = p0._required_text(
                raw_row.get("QUESTION_TITLE"),
                field="QUESTION_TITLE",
                maximum=20_000,
            )
            text = p0._required_text(
                raw_row.get("QUESTION_TEXT"),
                field="QUESTION_TEXT",
                maximum=100_000,
                allow_empty=True,
            )
            pool_raw = raw_row.get("DOC_IDS")
            if not isinstance(pool_raw, list):
                raise p0.TechqaP0QualificationError(
                    "DOC_IDS is not an array"
                )
            pool = tuple(
                p0._pool_identifier(value)
                for value in pool_raw
            )
            gold = p0._identifier(
                raw_row.get("DOCUMENT"), field="DOCUMENT"
            )
            # Parse offsets again only as an integrity check.  Their values
            # never enter selection, action formation, or scoring.
            start = p0._offset(
                raw_row.get("START_OFFSET"), field="START_OFFSET"
            )
            end = p0._offset(
                raw_row.get("END_OFFSET"), field="END_OFFSET"
            )
            if end <= start:
                raise p0.TechqaP0QualificationError(
                    "gold span is empty or reversed"
                )
            question = formal.VerifiedQuestion(
                question_id=question_id,
                question_title=title,
                question_text=text,
                document_ids=pool,
                gold_document_id=gold,
            )
            if (
                formal.operational_family(title, text)
                != identity.family
                or question.normalized_query_sha256
                != identity.normalized_query_sha256
            ):
                raise formal.TechqaP1FormalError(
                    "eligible question identity binding drifted"
                )
            answerable.append(question)
            seen_eligible.add(question_id)
        except (
            p0.TechqaP0QualificationError,
            formal.TechqaP1FormalError,
        ) as exc:
            raise TechqaP1RuntimeError(
                f"{split} answerable projection drifted after P0"
            ) from exc
    if seen_eligible != set(eligible):
        raise TechqaP1RuntimeError(
            f"{split} eligibility manifest/source coverage is not exact"
        )
    return tuple(answerable)


def _preselect_questions(
    *,
    training: Sequence[formal.VerifiedQuestion],
    dev: Sequence[formal.VerifiedQuestion],
    secret: bytes,
) -> tuple[
    tuple[formal.VerifiedQuestion, ...],
    tuple[formal.VerifiedQuestion, ...],
]:
    """Use the sole frozen selector, then return its bounded question union."""

    try:
        selection = formal.select_question_cohorts(
            tuple(training),
            tuple(dev),
            hmac_secret=secret,
        )
    except formal.TechqaP1FormalError as exc:
        raise TechqaP1RuntimeError(
            "frozen question-only selection failed"
        ) from exc
    chosen = {
        "TRAIN": [
            item.question
            for block in selection.blocks
            for item in block.items
            if item.split == "TRAIN"
        ],
        "DEV": [
            item.question
            for block in selection.blocks
            for item in block.items
            if item.split == "DEV"
        ],
    }
    result = tuple(
        tuple(sorted(chosen[split], key=lambda row: row.question_id))
        for split in ("TRAIN", "DEV")
    )
    if sum(map(len, result)) != MAX_SELECTED_QUESTION_COUNT:
        raise TechqaP1RuntimeError(
            "bounded selected question count drifted"
        )
    return result


KvItemsFactory = Callable[
    [Any],
    Iterable[tuple[str, object]],
]


@dataclass(frozen=True, slots=True)
class DocumentLoadReceipt:
    corpus_document_count: int
    candidate_reference_count: int
    selected_unique_document_count: int
    retained_unreferenced_document_count: int
    corpus_file_sha256: str

    def safe_payload(self) -> dict[str, object]:
        return {
            "candidate_reference_count": self.candidate_reference_count,
            "corpus_document_count": self.corpus_document_count,
            "corpus_file_sha256": self.corpus_file_sha256,
            "maximum_candidate_reference_count": (
                MAX_SELECTED_CANDIDATE_REFERENCES
            ),
            "maximum_selected_document_count": (
                MAX_SELECTED_DOCUMENT_COUNT
            ),
            "retained_unreferenced_document_count": (
                self.retained_unreferenced_document_count
            ),
            "selected_unique_document_count": (
                self.selected_unique_document_count
            ),
            "streaming_parser": "ijson_kvitems_root_object_v1",
        }


def _stream_selected_documents(
    *,
    binding: AssetBinding,
    needed_ids: set[str],
    candidate_reference_count: int,
    kvitems_factory: KvItemsFactory | None = None,
) -> tuple[tuple[formal.VerifiedDocument, ...], DocumentLoadReceipt]:
    metadata = binding.verify_metadata(
        field=p0.CORPUS_BASENAME,
        maximum_bytes=MAX_CORPUS_BYTES,
    )
    if (
        not needed_ids
        or len(needed_ids) > MAX_SELECTED_DOCUMENT_COUNT
        or candidate_reference_count > MAX_SELECTED_CANDIDATE_REFERENCES
    ):
        raise TechqaP1RuntimeError(
            "selected candidate document bound drifted"
        )
    missing = set(needed_ids)
    documents: list[formal.VerifiedDocument] = []
    seen_selected: set[str] = set()
    count = 0
    with _open_direct_read(
        binding.path, field=p0.CORPUS_BASENAME
    ) as source:
        hashing = p0._HashingReader(source)
        iterator = (
            p0._ijson_kvitems(hashing)
            if kvitems_factory is None
            else kvitems_factory(hashing)
        )
        try:
            for root_id, raw_document in iterator:
                count += 1
                if root_id not in needed_ids:
                    continue
                if root_id in seen_selected:
                    raise TechqaP1RuntimeError(
                        "selected corpus document ID is duplicated"
                    )
                if (
                    not isinstance(raw_document, Mapping)
                    or not p0.DOCUMENT_REQUIRED_KEYS
                    <= set(raw_document)
                ):
                    raise TechqaP1RuntimeError(
                        "selected corpus document schema drifted"
                    )
                try:
                    document_id = p0._identifier(
                        root_id, field="corpus root document ID"
                    )
                    embedded_id = p0._identifier(
                        raw_document.get("_id"), field="document _id"
                    )
                    title = p0._required_text(
                        raw_document.get("title"),
                        field="document title",
                        maximum=1_000_000,
                    )
                    text = p0._required_text(
                        raw_document.get("text"),
                        field="document text",
                        maximum=1_000_000,
                    )
                except p0.TechqaP0QualificationError as exc:
                    raise TechqaP1RuntimeError(
                        "selected corpus document value drifted"
                    ) from exc
                if embedded_id != document_id:
                    raise TechqaP1RuntimeError(
                        "selected corpus root and embedded IDs disagree"
                    )
                # Do not deduplicate equal serialized bytes here.  P0 binds
                # one-to-one referenced bytes, and the official adapter
                # remains the fail-closed authority if that contract drifts.
                documents.append(
                    formal.VerifiedDocument(
                        document_id=document_id,
                        title=title,
                        text=text,
                    )
                )
                seen_selected.add(document_id)
                missing.discard(document_id)
        except TechqaP1RuntimeError:
            raise
        except (OSError, p0.TechqaP0QualificationError) as exc:
            raise TechqaP1RuntimeError(
                "bounded corpus stream failed"
            ) from exc
        corpus_hash = hashing.sha256
        corpus_size = hashing.size
    if (
        corpus_size != metadata.st_size
        or not hmac.compare_digest(corpus_hash, binding.sha256)
        or missing
        or len(documents) != len(needed_ids)
    ):
        raise TechqaP1RuntimeError(
            "selected corpus coverage or full-file binding failed"
        )
    documents.sort(key=lambda row: row.document_id)
    receipt = DocumentLoadReceipt(
        corpus_document_count=count,
        candidate_reference_count=candidate_reference_count,
        selected_unique_document_count=len(documents),
        retained_unreferenced_document_count=0,
        corpus_file_sha256=corpus_hash,
    )
    return tuple(documents), receipt


def load_verified_source(
    config: FormalConfig,
    *,
    secret: bytes,
    eligibility: EligibilityManifest,
    kvitems_factory: KvItemsFactory | None = None,
) -> tuple[formal.VerifiedSource, DocumentLoadReceipt]:
    training_all = _parse_answerable_questions(
        binding=config.training_q_a,
        split="TRAIN",
        eligible=eligibility.rows_by_split["TRAIN"],
    )
    dev_all = _parse_answerable_questions(
        binding=config.dev_q_a,
        split="DEV",
        eligible=eligibility.rows_by_split["DEV"],
    )
    training, dev = _preselect_questions(
        training=training_all,
        dev=dev_all,
        secret=secret,
    )
    all_questions = training + dev
    candidate_reference_count = sum(
        len(row.document_ids) for row in all_questions
    )
    needed_ids = {
        document_id
        for question in all_questions
        for document_id in question.document_ids
    }
    documents, receipt = _stream_selected_documents(
        binding=config.training_dev_technotes,
        needed_ids=needed_ids,
        candidate_reference_count=candidate_reference_count,
        kvitems_factory=kvitems_factory,
    )
    source = formal.VerifiedSource(
        training_questions=training,
        dev_questions=dev,
        documents=documents,
        commitments=formal.SourceCommitments(
            training_q_a_sha256=config.training_q_a.sha256,
            dev_q_a_sha256=config.dev_q_a.sha256,
            training_dev_technotes_sha256=(
                config.training_dev_technotes.sha256
            ),
            qualification_receipt_sha256=(
                config.qualification_receipt.sha256
            ),
        ),
    )
    return source, receipt


def _prepared_selection_question_ids(
    prepared: formal.PreparedStudy,
) -> frozenset[str]:
    return frozenset(
        item.question.question_id
        for block in prepared.selection.blocks
        for item in block.items
    )


def prepare_study(
    source: formal.VerifiedSource,
    *,
    secret: bytes,
) -> formal.PreparedStudy:
    """Call the frozen formal preparation and verify bounded preselection."""

    prepared = formal.prepare_formal_study(
        source, hmac_secret=secret
    )
    source_ids = frozenset(
        row.question_id
        for row in source.training_questions + source.dev_questions
    )
    if _prepared_selection_question_ids(prepared) != source_ids:
        raise TechqaP1RuntimeError(
            "authoritative selection diverged from bounded preselection"
        )
    return prepared


@dataclass(frozen=True, slots=True)
class PublicClusterBundle:
    stage: str
    cluster_index: int
    payload: Mapping[str, object]
    work_ids: tuple[str, ...]


def public_cluster_bundles(
    stage: formal.PreparedStage,
) -> tuple[PublicClusterBundle, ...]:
    if stage.block != formal.A_HOLD:
        raise TechqaP1RuntimeError(
            "official HippoRAG requested outside A_hold"
        )
    bundles: list[PublicClusterBundle] = []
    requests = stage.hippo_cluster_request_by_index
    for cluster in stage.clusters:
        request = requests.get(cluster.cluster_index)
        if request is None:
            raise TechqaP1RuntimeError(
                "frozen official cluster request disappeared"
            )
        payload = request.adapter_input
        checked = hippo_adapter.validate_input(payload)
        if checked.cluster_ordinal != cluster.cluster_index:
            raise TechqaP1RuntimeError(
                "official public cluster ordinal drifted"
            )
        # The formal controller formed this exact adapter input.  Re-check
        # only its public schema here; never reconstruct or transform it.
        bundles.append(
            PublicClusterBundle(
                stage=stage.block,
                cluster_index=cluster.cluster_index,
                payload=payload,
                work_ids=tuple(
                    row.work_id for row in request.query_bindings
                ),
            )
        )
    if (
        len(bundles) != HIPPO_CLUSTERS_PER_STAGE
        or tuple(row.cluster_index for row in bundles)
        != tuple(range(HIPPO_CLUSTERS_PER_STAGE))
    ):
        raise TechqaP1RuntimeError(
            "official stage cluster registry drifted"
        )
    return tuple(bundles)


class HippoLauncher(Protocol):
    def __call__(
        self,
        *,
        public_input: Mapping[str, object],
        cluster_root: Path,
        gpu_id: str,
    ) -> Mapping[str, object]:
        """Execute one fresh cluster once and return its safe adapter output."""


class _ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[bytes]] = set()

    def add(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes.add(process)
            if len(self._processes) > HIPPO_MAX_CONCURRENCY:
                raise TechqaP1RuntimeError(
                    "official process concurrency cap drifted"
                )

    def discard(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes.discard(process)

    def cancel_all(self) -> None:
        with self._lock:
            processes = tuple(self._processes)
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


class ProductionHippoLauncher:
    """Thin reuse boundary around the frozen official cluster adapter."""

    def __init__(self, assets: RuntimeAssets) -> None:
        self.assets = assets
        self.registry = _ProcessRegistry()

    def cancel_all(self) -> None:
        self.registry.cancel_all()

    def __call__(
        self,
        *,
        public_input: Mapping[str, object],
        cluster_root: Path,
        gpu_id: str,
    ) -> Mapping[str, object]:
        if gpu_id not in GPU_IDS:
            raise TechqaP1RuntimeError("physical GPU binding drifted")
        cluster = hippo_adapter.validate_input(public_input)
        _private_directory(cluster_root, fresh=True)
        custody = cluster_root / "custody"
        scratch = cluster_root / "scratch"
        _private_directory(custody, fresh=True)
        _private_directory(scratch, fresh=True)
        for name in ("cache", "home", "tmp", "model_aliases"):
            _private_directory(scratch / name, fresh=True)

        input_path = scratch / "cluster.input.private.json"
        output_path = scratch / "cluster.output.private.json"
        adapter_work_root = scratch / "adapter_work"
        input_file_sha = _write_once(
            input_path,
            hippo_adapter.canonical_bytes(
                public_input, newline=True
            ),
            mode=PRIVATE_FILE_MODE,
        )
        alias_root = scratch / "model_aliases"
        base_runtime._short_alias(
            alias_root,
            base_runtime.LLM_ALIAS,
            self.assets.paths.smollm_model_root,
        )
        base_runtime._short_alias(
            alias_root,
            base_runtime.MINILM_ALIAS,
            self.assets.paths.minilm_model_root,
        )
        network_path = custody / "network.private.strace"
        stdout_path = custody / "worker.stdout.private.bin"
        stderr_path = custody / "worker.stderr.private.bin"
        worker_command = [
            self.assets.paths.official_python,
            "-S",
            "-B",
            "-m",
            OFFICIAL_ADAPTER_MODULE,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--work-root",
            str(adapter_work_root),
            "--llm-model",
            base_runtime.LLM_ALIAS,
            "--embedding-model",
            base_runtime.MINILM_ALIAS,
            "--hipporag-source-root",
            self.assets.paths.hipporag_source_root,
            "--project-root",
            self.assets.paths.project_root,
        ]
        if any(
            absolute in worker_command
            for absolute in (
                self.assets.paths.smollm_model_root,
                self.assets.paths.minilm_model_root,
            )
        ):
            raise TechqaP1RuntimeError(
                "absolute model path escaped into worker argv"
            )
        command = [
            self.assets.paths.strace_path,
            "-f",
            "-qq",
            "-e",
            "trace=socket,connect",
            "-o",
            str(network_path),
            *worker_command,
        ]
        environment = base_runtime._child_environment(
            paths=self.assets.paths,
            scratch=scratch,
            physical_gpu=gpu_id,
            pythonpath=self.assets.paths.official_pythonpath(),
            python=self.assets.paths.official_python,
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        out_descriptor = os.open(stdout_path, flags, PRIVATE_FILE_MODE)
        err_descriptor = os.open(stderr_path, flags, PRIVATE_FILE_MODE)
        try:
            with os.fdopen(out_descriptor, "wb") as stdout, os.fdopen(
                err_descriptor, "wb"
            ) as stderr:
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=alias_root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                except OSError as exc:
                    raise TechqaP1RuntimeError(
                        "official cluster launch failed; retry is forbidden"
                    ) from exc
                self.registry.add(process)
                try:
                    try:
                        returncode = process.wait(
                            timeout=self.assets.timeout_seconds
                        )
                    except subprocess.TimeoutExpired as exc:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise TechqaP1RuntimeError(
                            "official cluster timed out; retry is forbidden"
                        ) from exc
                finally:
                    self.registry.discard(process)
                    stdout.flush()
                    stderr.flush()
                    os.fsync(stdout.fileno())
                    os.fsync(stderr.fileno())
        finally:
            # File descriptors are owned by the context managers if fdopen
            # succeeded; on an earlier failure close any still-open handles.
            for descriptor in (out_descriptor, err_descriptor):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        for log_path in (stdout_path, stderr_path):
            if log_path.stat().st_size > MAX_WORKER_LOG_BYTES:
                raise TechqaP1RuntimeError(
                    "official worker log exceeded its private bound"
                )
        if returncode != 0:
            raise TechqaP1RuntimeError(
                "official cluster exited nonzero; retry is forbidden"
            )
        network = base_runtime._network_audit(network_path)
        output = _read_canonical_json(
            output_path,
            field="official cluster output",
            mode=PRIVATE_FILE_MODE,
            maximum_bytes=32 * 1024 * 1024,
        )
        checked = hippo_adapter.validate_output(
            output, expected_input=public_input
        )
        launch_receipt = self_hashed(
            {
                "cluster_ordinal": cluster.cluster_ordinal,
                "gpu_id": gpu_id,
                "input_file_sha256": input_file_sha,
                "network_syscall_audit": network,
                "online_or_API_evaluator_call_count": 0,
                "output_self_sha256": checked["self_sha256"],
                "retry_replay_resample_count": 0,
                "schema": LAUNCH_RECEIPT_SCHEMA,
                "stage": cluster.stage,
                "stderr_sha256": base_runtime._sha256_file(
                    stderr_path
                ),
                "stdout_sha256": base_runtime._sha256_file(
                    stdout_path
                ),
                "study_id": formal.STUDY_ID,
            }
        )
        _write_json_once(
            custody / "launch.private.json",
            launch_receipt,
            mode=FROZEN_FILE_MODE,
        )
        return checked


@dataclass(frozen=True, slots=True)
class StageHippoRun:
    stage: str
    outputs: tuple[Mapping[str, object], ...]
    public_inputs: tuple[Mapping[str, object], ...]
    work_ids: tuple[tuple[str, ...], ...]

    def safe_payload(self) -> dict[str, object]:
        return {
            "cluster_count": len(self.outputs),
            "cluster_output_self_sha256": [
                str(row["self_sha256"]) for row in self.outputs
            ],
            "gpu_schedule": [
                {"cluster_ordinal": cluster, "gpu_id": GPU_IDS[index]}
                for wave in HIPPO_WAVES
                for index, cluster in enumerate(wave)
            ],
            "maximum_concurrent_processes": HIPPO_MAX_CONCURRENCY,
            "retry_replay_resample_count": 0,
            "stage": self.stage,
            "wave_count": len(HIPPO_WAVES),
        }


def run_hippo_stage(
    stage: formal.PreparedStage,
    *,
    launcher: HippoLauncher,
    stage_root: Path,
) -> StageHippoRun:
    """Run four clusters in exactly two waves with one process per GPU."""

    bundles = public_cluster_bundles(stage)
    _private_directory(stage_root, fresh=True)
    outputs: dict[int, Mapping[str, object]] = {}
    for wave_index, wave in enumerate(HIPPO_WAVES):
        with ThreadPoolExecutor(
            max_workers=HIPPO_MAX_CONCURRENCY,
            thread_name_prefix=(
                f"techqa-{stage.block}-hippo-wave-{wave_index}"
            ),
        ) as pool:
            futures = {
                cluster_index: pool.submit(
                    launcher,
                    public_input=bundles[cluster_index].payload,
                    cluster_root=(
                        stage_root / f"cluster_{cluster_index}"
                    ),
                    gpu_id=GPU_IDS[position],
                )
                for position, cluster_index in enumerate(wave)
            }
            try:
                for cluster_index in wave:
                    raw = futures[cluster_index].result()
                    outputs[cluster_index] = (
                        hippo_adapter.validate_output(
                            raw,
                            expected_input=(
                                bundles[cluster_index].payload
                            ),
                        )
                    )
            except BaseException:
                cancel = getattr(launcher, "cancel_all", None)
                if callable(cancel):
                    cancel()
                raise
    ordered = tuple(outputs[index] for index in range(4))
    return StageHippoRun(
        stage=stage.block,
        outputs=ordered,
        public_inputs=tuple(row.payload for row in bundles),
        work_ids=tuple(row.work_ids for row in bundles),
    )


def _formal_hippo_cluster_runs(
    prepared: formal.PreparedStudy,
    run: StageHippoRun,
) -> tuple[formal.OfficialHippoClusterRun, ...]:
    if run.stage != formal.A_HOLD:
        raise TechqaP1RuntimeError(
            "formal Hippo binding received a non-A_hold stage"
        )
    expected_inputs = tuple(
        request.adapter_input
        for request in prepared.hippo_cluster_requests
    )
    if run.public_inputs != expected_inputs:
        raise TechqaP1RuntimeError(
            "executed Hippo inputs diverged from frozen cluster requests"
        )
    return tuple(
        formal.OfficialHippoClusterRun(
            adapter_input=public_input,
            safe_output=output,
        )
        for public_input, output in zip(
            run.public_inputs, run.outputs, strict=True
        )
    )


FinalizeCallable = Callable[
    [Sequence[formal.OfficialHippoClusterRun]],
    formal.FormalResult,
]


@dataclass(frozen=True, slots=True)
class ExecutedStudy:
    result: formal.FormalResult
    a_hold_hippo: StageHippoRun


def execute_prepared_study(
    prepared: formal.PreparedStudy,
    *,
    launcher: HippoLauncher,
    execution_root: Path,
    finalize: FinalizeCallable | None = None,
) -> ExecutedStudy:
    """Execute A_hold then apply the frozen promotion/L5 lifecycle."""

    _private_directory(execution_root, fresh=True)
    a_hold = run_hippo_stage(
        prepared.a_hold,
        launcher=launcher,
        stage_root=execution_root / formal.A_HOLD,
    )
    controller = formal.OneShotFormalController(prepared)
    finalizer = controller.finalize if finalize is None else finalize
    result = finalizer(_formal_hippo_cluster_runs(prepared, a_hold))
    if not isinstance(result, formal.FormalResult):
        raise TechqaP1RuntimeError("formal finalizer result drifted")
    if result.m_search is not None:
        if (
            result.safe_terminal.get("A_hold", {}).get(
                "promotion_passed"
            )
            is not True
        ):
            raise TechqaP1RuntimeError(
                "M_search appeared without frozen promotion"
            )
    elif (
        result.safe_terminal.get("M_search", {}).get(
            "actions_materialized_after_promotion"
        )
        is not False
    ):
        raise TechqaP1RuntimeError(
            "unpromoted M_search terminal binding drifted"
        )
    return ExecutedStudy(
        result=result,
        a_hold_hippo=a_hold,
    )


def _runtime_private_archive(
    *,
    prepared: formal.PreparedStudy,
    executed: ExecutedStudy,
    attempt_file_sha256: str,
    source_load: DocumentLoadReceipt,
    eligibility: EligibilityManifest,
) -> dict[str, object]:
    return {
        "attempt_file_sha256": attempt_file_sha256,
        "formal_private_archive": dict(executed.result.private_archive),
        "formal_private_archive_sha256": (
            executed.result.private_archive_sha256
        ),
        "eligibility_manifest_file_sha256": (
            eligibility.file_sha256
        ),
        "eligibility_manifest_self_sha256": (
            eligibility.self_sha256
        ),
        "official_HippoRAG": {
            formal.A_HOLD: {
                "public_cluster_inputs": [
                    dict(row)
                    for row in executed.a_hold_hippo.public_inputs
                ],
                "safe_cluster_outputs": [
                    dict(row) for row in executed.a_hold_hippo.outputs
                ],
                "work_ids_by_cluster": [
                    list(row) for row in executed.a_hold_hippo.work_ids
                ],
            },
            formal.M_SEARCH: None,
        },
        "prepared_prepromotion_archive_sha256": (
            prepared.prepromotion_archive_sha256
        ),
        "schema": PRIVATE_ARCHIVE_SCHEMA,
        "source_load": source_load.safe_payload(),
        "study_id": formal.STUDY_ID,
    }


def _safe_terminal(
    *,
    config: FormalConfig,
    executed: ExecutedStudy,
    attempt_file_sha256: str,
    private_archive_file_sha256: str,
    private_archive_sha256: str,
    source_load: DocumentLoadReceipt,
    eligibility: EligibilityManifest,
) -> dict[str, object]:
    return self_hashed(
        {
            "aggregate_only_public_terminal": True,
            "attempt_count": 1,
            "attempt_file_sha256": attempt_file_sha256,
            "config_self_sha256": config.config_self_sha256,
            "formal_safe_terminal": dict(
                executed.result.safe_terminal
            ),
            "eligibility_manifest_file_sha256": (
                eligibility.file_sha256
            ),
            "eligibility_manifest_self_sha256": (
                eligibility.self_sha256
            ),
            "implementation_freeze_sha256": (
                config.implementation_freeze.sha256
            ),
            "item_query_document_qrel_action_values_published": False,
            "official_HippoRAG": {
                formal.A_HOLD: executed.a_hold_hippo.safe_payload(),
                formal.M_SEARCH: None,
            },
            "online_or_API_evaluator_call_count": 0,
            "private_archive_file_sha256": (
                private_archive_file_sha256
            ),
            "private_archive_sha256": private_archive_sha256,
            "qualification_receipt_sha256": (
                config.qualification_receipt.sha256
            ),
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": SAFE_TERMINAL_SCHEMA,
            "source_load": source_load.safe_payload(),
            "status": "terminal_complete_once",
            "study_id": formal.STUDY_ID,
        }
    )


def _write_failure_terminal(
    *,
    config: FormalConfig,
    attempt_file_sha256: str,
    failure_stage: str,
    error: BaseException,
) -> dict[str, object]:
    if _SAFE_STAGE.fullmatch(failure_stage) is None:
        failure_stage = "internal_stage"
    private = self_hashed(
        {
            "attempt_file_sha256": attempt_file_sha256,
            "error_class": type(error).__name__,
            "error_message": str(error),
            "failure_stage": failure_stage,
            "schema": f"{VERSION}_private_failure_v1",
            "study_id": formal.STUDY_ID,
        }
    )
    private_file_sha = _write_json_once(
        config.work_root / "failure.private.json",
        private,
        mode=PRIVATE_FILE_MODE,
    )
    terminal = self_hashed(
        {
            "aggregate_only_public_terminal": True,
            "attempt_count": 1,
            "attempt_file_sha256": attempt_file_sha256,
            "failure_stage": failure_stage,
            "item_query_document_qrel_action_values_published": False,
            "online_or_API_evaluator_call_count": 0,
            "private_failure_file_sha256": private_file_sha,
            "retry_or_replay_authorized": False,
            "schema": FAILURE_TERMINAL_SCHEMA,
            "status": "terminal_failed_once_no_retry",
            "study_id": formal.STUDY_ID,
        }
    )
    _write_json_once(
        config.work_root / "formal_terminal.json",
        terminal,
        mode=SAFE_TERMINAL_MODE,
    )
    return terminal


def run_formal_once(
    *,
    config_path: Path,
    launcher: HippoLauncher | None = None,
    kvitems_factory: KvItemsFactory | None = None,
) -> dict[str, object]:
    """Run the complete formal lifecycle once.

    ``launcher`` and ``kvitems_factory`` are injectable only for source-free
    and synthetic tests.  Production passes neither.
    """

    config = load_config(config_path)
    qualification, _implementation = _verify_supporting_receipts(
        config
    )
    attempt_file_sha = claim_formal_attempt(config)
    stage = "eligibility_load"
    try:
        eligibility = _load_eligibility_manifest(
            config,
            qualification_receipt=qualification,
        )
        stage = "post_attempt_secret"
        secret = _load_secret(config.hmac_secret)
        stage = "source_load"
        source, source_load = load_verified_source(
            config,
            secret=secret,
            eligibility=eligibility,
            kvitems_factory=kvitems_factory,
        )
        stage = "formal_prepare"
        prepared = prepare_study(source, secret=secret)
        _write_json_once(
            config.work_root / "prepromotion.private.json",
            prepared.prepromotion_private_payload(),
            mode=PRIVATE_FILE_MODE,
        )
        stage = "official_and_finalize"
        actual_launcher = (
            ProductionHippoLauncher(config.runtime)
            if launcher is None
            else launcher
        )
        executed = execute_prepared_study(
            prepared,
            launcher=actual_launcher,
            execution_root=config.work_root / "formal_execution",
        )
        stage = "archive_finalize"
        private_archive = _runtime_private_archive(
            prepared=prepared,
            executed=executed,
            attempt_file_sha256=attempt_file_sha,
            source_load=source_load,
            eligibility=eligibility,
        )
        private_hash = stable_hash(private_archive)
        private_file_hash = _write_json_once(
            config.work_root / "formal.private.json",
            private_archive,
            mode=PRIVATE_FILE_MODE,
        )
        terminal = _safe_terminal(
            config=config,
            executed=executed,
            attempt_file_sha256=attempt_file_sha,
            private_archive_file_sha256=private_file_hash,
            private_archive_sha256=private_hash,
            source_load=source_load,
            eligibility=eligibility,
        )
        _write_json_once(
            config.work_root / "formal_terminal.json",
            terminal,
            mode=SAFE_TERMINAL_MODE,
        )
        return terminal
    except BaseException as exc:
        return _write_failure_terminal(
            config=config,
            attempt_file_sha256=attempt_file_sha,
            failure_stage=stage,
            error=exc,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = run_formal_once(config_path=arguments.config)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "self_sha256": result["self_sha256"],
                "status": result["status"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return int(result["status"] != "terminal_complete_once")


__all__ = [
    "ALLOWED_ASSET_MODES",
    "ATTEMPT_SCHEMA",
    "AssetBinding",
    "CONFIG_MODE",
    "CONFIG_SCHEMA",
    "DEFAULT_HIPPO_TIMEOUT_SECONDS",
    "DocumentLoadReceipt",
    "EligibilityManifest",
    "EligibleQuestionIdentity",
    "ExecutedStudy",
    "FAILURE_TERMINAL_SCHEMA",
    "FormalConfig",
    "GPU_IDS",
    "HIPPO_CLUSTERS_PER_STAGE",
    "HIPPO_MAX_CONCURRENCY",
    "HIPPO_WAVES",
    "HippoLauncher",
    "MAX_SELECTED_CANDIDATE_REFERENCES",
    "MAX_SELECTED_DOCUMENT_COUNT",
    "MAX_SELECTED_QUESTION_COUNT",
    "P0_ELIGIBILITY_RULE_VERSION",
    "P0_ELIGIBILITY_SCHEMA",
    "PRIVATE_ARCHIVE_SCHEMA",
    "ProductionHippoLauncher",
    "PublicClusterBundle",
    "RuntimeAssets",
    "SAFE_TERMINAL_SCHEMA",
    "StageHippoRun",
    "TechqaP1RuntimeError",
    "canonical_bytes",
    "claim_formal_attempt",
    "execute_prepared_study",
    "load_config",
    "load_verified_source",
    "main",
    "prepare_study",
    "public_cluster_bundles",
    "run_formal_once",
    "run_hippo_stage",
    "self_hashed",
    "stable_hash",
]


if __name__ == "__main__":
    raise SystemExit(main())
