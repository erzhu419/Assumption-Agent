"""One-shot homologous three-arm MuSiQue development runner.

The worker plan and per-item inputs are deliberately gold-free.  Retrieval and
all eighteen generator calls finish before the private answer/support index is
opened.  Formal execution fixes the provider transport and official HippoRAG
adapter in code; the separately named synthetic entry point is the only place
where test dependencies may be injected.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence

# Support the audited direct-file CLI from an arbitrary working directory.
if __package__ in {None, ""}:  # pragma: no cover - exercised by subprocess test
    _BOOTSTRAP_ROOT = Path(__file__).resolve(strict=True).parents[2]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
    read_hashed_json_v2,
)
from replication_runtime.noaa_gsod_v1.development_runner import (
    ModelRequest,
    ModelTransport,
    ProviderCredential,
    ProviderProtocolError,
    ProviderTransportUnavailable,
    UrllibOpenAICompatibleTransport,
    load_provider_credential,
)

from assumption_agent.benchmarks.musique_official_core_comparison_v1 import (
    evaluate_aliases_primary,
    evaluate_aliases_secondary,
    evaluate_support_primary,
    evaluate_support_secondary,
)
from assumption_agent.benchmarks.musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    load_formation_receipt,
    load_frozen_program,
    retrieve as typed_retrieve,
)


RUNNER_VERSION = "musique_homologous_three_arm_formal_runner_v1"
WORKER_PLAN_VERSION = "musique_homologous_three_arm_worker_plan_v1"
ITEM_INPUT_VERSION = "musique_homologous_three_arm_gold_free_input_v1"
PRIVATE_INDEX_VERSION = "musique_homologous_three_arm_private_index_v1"
PROVIDER_PRECOMMIT_VERSION = "musique_three_arm_provider_precommit_v1"
PROVIDER_SELECTION_VERSION = "musique_three_arm_provider_selection_v1"
LAUNCH_VERSION = "musique_three_arm_generation_launch_v1"
CLAIM_VERSION = "musique_three_arm_generation_claim_v1"
CLAIM_SET_VERSION = "musique_three_arm_generation_claim_set_v1"
TERMINAL_VERSION = "musique_three_arm_generation_terminal_v1"
PRIVATE_EVALUATION_VERSION = "musique_three_arm_private_offline_evaluation_v1"
PUBLIC_REPORT_VERSION = "musique_three_arm_public_aggregate_report_v1"
PUBLIC_FREEZE_VERSION = "musique_homologous_three_arm_public_freeze_v2"
CANARY_VERSION = "musique_three_arm_constant_transport_canary_v1"
CONTEXT_SERIALIZATION_VERSION = "musique_title_text_context_serialization_v1"
PROMPT_VERSION = "musique_closed_context_answer_json_prompt_v1"
OFFICIAL_ADAPTER_ID = "musique_official_hipporag_retrieve_only_v1"
OFFICIAL_ADAPTER_MODULE = (
    "replication_runtime.musique_official_hipporag_v1.adapter"
)
OFFICIAL_ADAPTER_FUNCTION = "run_official_hipporag_retrieve_only"
OFFICIAL_BINDING_RECEIPT_RELATIVE = (
    "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
)
FORMAL_CUSTODY_RECEIPT_RELATIVE = (
    "manifests/musique_development_gold_safe_custody_receipt_v2.json"
)
FORMAL_PRIVATE_INDEX_BINDING_RELATIVE = (
    "manifests/musique_development_private_index_public_binding_v1.json"
)
FORMAL_PUBLIC_FREEZE_RELATIVE = (
    "manifests/musique_formal_development_pre_run_freeze_v2.json"
)
OFFICIAL_HIPPORAG_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
FORMAL_CUSTODY_RECEIPT_RELATIVE = (
    "manifests/musique_development_gold_safe_custody_receipt_v2.json"
)
FORMAL_PRIVATE_INDEX_BINDING_RELATIVE = (
    "manifests/musique_development_private_index_public_binding_v1.json"
)
FORMAL_PUBLIC_FREEZE_RELATIVE = (
    "manifests/musique_formal_development_pre_run_freeze_v2.json"
)
MODEL_ID = "gpt-5.4-mini"
PROVIDER_ORIGIN = "https://ruoli.dev"
TOP_K = 5
ITEM_COUNT = 6
ARM_IDS = (
    "canonical_order_top_k_context_baseline",
    "assumption_retrieval",
    "official_hipporag_retrieval",
)
WORK_UNIT_COUNT = ITEM_COUNT * len(ARM_IDS)
MODEL_REQUEST_BODY_BYTE_BUDGET = 64 * 1024
MODEL_OUTPUT_TOKEN_BUDGET = 256
MAXIMUM_MODEL_CONCURRENCY = WORK_UNIT_COUNT
EXECUTION_ROOT_RELATIVE_PATH = "formal_execution"
CONSUMPTION_MARKER_RELATIVE_PATH = "execution.authorization.consumed.json"
PRIVATE_INDEX_RELATIVE_PATH = "private_index.json"
STUDY_ID = "musique_homologous_three_arm_development_v1"
RUNNER_IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/benchmarks/musique_three_arm_formal_runner_v1.py",
    "assumption_agent/benchmarks/musique_development_custody_v1.py",
    "assumption_agent/benchmarks/musique_development_freeze_v1.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/__init__.py",
    "replication_runtime/financial_semantic_v2/__init__.py",
    "replication_runtime/financial_semantic_v2/durable_state.py",
    "replication_runtime/noaa_gsod_v1/__init__.py",
    "replication_runtime/noaa_gsod_v1/acquire.py",
    "replication_runtime/noaa_gsod_v1/contract.py",
    "replication_runtime/noaa_gsod_v1/development_freeze.py",
    "replication_runtime/noaa_gsod_v1/development_implementation.py",
    "replication_runtime/noaa_gsod_v1/development_runner.py",
    "replication_runtime/noaa_gsod_v1/development_schemas.py",
    "replication_runtime/noaa_gsod_v1/development_source.py",
    "replication_runtime/noaa_gsod_v1/oracle_sqlite.py",
    "replication_runtime/noaa_gsod_v1/oracle_stdlib.py",
    "replication_runtime/noaa_gsod_v1/pack.py",
    "replication_runtime/noaa_gsod_v1/schemas.py",
    "replication_runtime/noaa_gsod_v1/train_export.py",
    "replication_runtime/noaa_gsod_v1/train_schemas.py",
    "replication_runtime/noaa_gsod_v1/typed_relational.py",
    "replication_runtime/musique_official_hipporag_v1/__init__.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)

WORKER_PLAN_FILENAME = "worker_plan.json"
PRIVATE_INDEX_FILENAME = "private_index.json"
PROVIDER_PRECOMMIT_FILENAME = "provider.identity.precommit.json"
PROVIDER_SELECTION_FILENAME = "provider.selection.json"
LAUNCH_FILENAME = "generation.launch.json"
CLAIM_SET_FILENAME = "generation.claim-set.json"
PRIVATE_EVALUATION_FILENAME = "evaluation.private.json"
PUBLIC_REPORT_FILENAME = "report.public.json"

GENERATOR_SYSTEM_PROMPT = (
    "Answer the question using only the supplied ordered documents. Return "
    "exactly one canonical JSON object with the sole string field answer. "
    "Do not use tools, files, search, or outside knowledge."
)
CANARY_SYSTEM_PROMPT = (
    "This is a fixed transport-only availability canary. Do not use tools, "
    "files, network search, or external context. Return any non-empty response."
)
CANARY_PAYLOAD: dict[str, Any] = {
    "canary_version": CANARY_VERSION,
    "contains_task_or_development_content": False,
    "fixed_probe": "transport_complete_response_only",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ANONYMOUS_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "accepted_aliases",
        "answer",
        "answers",
        "context",
        "corpus",
        "documents",
        "item_id",
        "paragraph_text",
        "prediction",
        "question",
        "support_indices",
        "text",
        "title",
    }
)


class MuSiQueFormalRunnerError(RuntimeError):
    """The one-shot formal boundary failed closed."""


class MuSiQueNoReplayError(MuSiQueFormalRunnerError):
    """A durable authorization was consumed and may not be replayed."""


class MuSiQueOutputContractError(MuSiQueFormalRunnerError):
    """A generator output violated the frozen answer contract."""


class OfficialRetrieveOnly(Protocol):
    def __call__(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]: ...


def _project_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MuSiQueFormalRunnerError(f"{field} must be lowercase sha256")
    return value


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: str | Path, field: str) -> Path:
    candidate = _absolute_lexical(path)
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueFormalRunnerError(f"{field} contains a symbolic-link component")
    return candidate


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def execution_root_commitment(
    *, authorization_id: str, development_root_sha256: str
) -> str:
    return stable_hash(
        {
            "authorization_id": _require_sha256(authorization_id, "authorization id"),
            "development_root_commitment": _require_sha256(
                development_root_sha256, "development root commitment"
            ),
            "execution_root_relative_path": EXECUTION_ROOT_RELATIVE_PATH,
        }
    )


def current_runner_implementation_binding(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    rows = []
    for relative in RUNNER_IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueFormalRunnerError(f"runner implementation missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "files": rows,
        "schema": "musique_three_arm_runner_implementation_set_v1",
        "set_sha256": stable_hash(rows),
    }


def provider_identity_binding(
    *, plus_channel_id: str, pro_channel_id: str
) -> dict[str, Any]:
    plus = str(plus_channel_id).strip().casefold()
    pro = str(pro_channel_id).strip().casefold()
    if (
        _ANONYMOUS_ID.fullmatch(plus) is None
        or _ANONYMOUS_ID.fullmatch(pro) is None
        or "plus" not in plus
        or "pro" not in pro
        or plus == pro
    ):
        raise MuSiQueFormalRunnerError("provider channel identities are invalid")
    body = {
        "api_origin": PROVIDER_ORIGIN,
        "model": MODEL_ID,
        "plus_channel_id": plus,
        "pro_channel_id": pro,
    }
    return {**body, "identity_sha256": stable_hash(body)}


def _read_hashed_json(path: Path, *, hash_field: str) -> dict[str, Any]:
    try:
        return read_hashed_json_v2(path, hash_field=hash_field)
    except Exception as exc:
        raise MuSiQueFormalRunnerError(f"invalid hashed JSON: {path.name}") from exc


def _safe_relative_file(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise MuSiQueFormalRunnerError(f"{field} is not a safe relative path")
    lexical = Path(relative)
    if lexical.is_absolute() or any(part in {"", ".", ".."} for part in lexical.parts):
        raise MuSiQueFormalRunnerError(f"{field} is not a safe relative path")
    path = root.joinpath(*lexical.parts)
    current = root
    for component in lexical.parts:
        current = current / component
        if current.is_symlink():
            raise MuSiQueFormalRunnerError(f"{field} contains a symbolic-link component")
    if not path.is_file():
        raise MuSiQueFormalRunnerError(f"{field} is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MuSiQueFormalRunnerError(f"{field} escapes the development root") from exc
    return resolved


def _require_ignored_or_external(path: Path) -> None:
    repository = _project_root()
    try:
        relative = path.resolve(strict=False).relative_to(repository)
    except ValueError:
        return
    completed = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--no-index", "-q", "--", relative.as_posix()],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise MuSiQueFormalRunnerError("private execution output must be git-ignored")


def _validate_worker_plan(
    path: Path,
    *,
    development_root: Path,
    plus_channel_id: str,
    pro_channel_id: str,
) -> dict[str, Any]:
    worker = _read_hashed_json(path, hash_field="worker_plan_hash")
    expected_keys = {
        "authorization_id",
        "arm_ids",
        "consumption_marker_relative_path",
        "custody_binding",
        "development_root_commitment",
        "execution_root_commitment",
        "execution_root_relative_path",
        "formal_official_binding",
        "item_count",
        "items",
        "private_index_binding",
        "provider_identity",
        "runner_implementation",
        "shared_contract",
        "study_id",
        "typed_binding",
        "worker_plan_hash",
        "worker_plan_version",
    }
    if set(worker) != expected_keys:
        raise MuSiQueFormalRunnerError("worker plan schema drifted")
    authorization_id = _require_sha256(worker.get("authorization_id"), "authorization id")
    frozen_development_commitment = _require_sha256(
        worker.get("development_root_commitment"), "development root commitment"
    )
    if (
        worker.get("worker_plan_version") != WORKER_PLAN_VERSION
        or worker.get("study_id") != STUDY_ID
        or worker.get("item_count") != ITEM_COUNT
        or worker.get("arm_ids") != list(ARM_IDS)
        or worker.get("execution_root_relative_path")
        != EXECUTION_ROOT_RELATIVE_PATH
        or worker.get("consumption_marker_relative_path")
        != CONSUMPTION_MARKER_RELATIVE_PATH
        or worker.get("execution_root_commitment")
        != execution_root_commitment(
            authorization_id=authorization_id,
            development_root_sha256=frozen_development_commitment,
        )
        or worker.get("provider_identity")
        != provider_identity_binding(
            plus_channel_id=plus_channel_id, pro_channel_id=pro_channel_id
        )
    ):
        raise MuSiQueFormalRunnerError("worker plan frozen identity drifted")
    implementation = worker.get("runner_implementation")
    if not isinstance(implementation, Mapping) or implementation != current_runner_implementation_binding():
        raise MuSiQueFormalRunnerError("live runner implementation drifted")
    shared = worker.get("shared_contract")
    expected_shared = {
        "context_serialization_version": CONTEXT_SERIALIZATION_VERSION,
        "generator_prompt_version": PROMPT_VERSION,
        "maximum_model_concurrency": MAXIMUM_MODEL_CONCURRENCY,
        "model": MODEL_ID,
        "model_output_token_budget": MODEL_OUTPUT_TOKEN_BUDGET,
        "model_request_body_byte_budget": MODEL_REQUEST_BODY_BYTE_BUDGET,
        "overflow_policy": "fail_closed_no_truncation",
        "online_evaluator_calls": 0,
        "replays": 0,
        "resamples": 0,
        "retries": 0,
        "temperature": 0,
        "top_k": TOP_K,
        "work_unit_count": WORK_UNIT_COUNT,
    }
    if shared != expected_shared:
        raise MuSiQueFormalRunnerError("shared three-arm contract drifted")
    typed = worker.get("typed_binding")
    if not isinstance(typed, Mapping) or set(typed) != {
        "formation_receipt_relative_path",
        "formation_receipt_file_sha256",
        "formation_receipt_hash",
        "frozen_program_relative_path",
        "frozen_program_file_sha256",
        "frozen_program_hash",
    }:
        raise MuSiQueFormalRunnerError("typed retriever binding drifted")
    for prefix in ("formation_receipt", "frozen_program"):
        bound_path = _safe_relative_file(
            development_root, typed[f"{prefix}_relative_path"], f"typed {prefix}"
        )
        if _sha256_file(bound_path) != _require_sha256(
            typed[f"{prefix}_file_sha256"], f"typed {prefix} hash"
        ):
            raise MuSiQueFormalRunnerError("typed retriever file hash drifted")
    typed_receipt_path = _safe_relative_file(
        development_root,
        typed["formation_receipt_relative_path"],
        "typed formation receipt",
    )
    typed_program_path = _safe_relative_file(
        development_root,
        typed["frozen_program_relative_path"],
        "typed frozen program",
    )
    typed_receipt = load_formation_receipt(typed_receipt_path)
    typed_program = load_frozen_program(
        typed_program_path,
        receipt_path=typed_receipt_path,
        verify_live=True,
    )
    if (
        typed_receipt.get("receipt_hash") != typed.get("formation_receipt_hash")
        or typed_program.program_hash != typed.get("frozen_program_hash")
    ):
        raise MuSiQueFormalRunnerError("typed receipt/program semantic binding drifted")
    official = worker.get("formal_official_binding")
    if not isinstance(official, Mapping) or set(official) != {
        "adapter_id",
        "binding_receipt_file_sha256",
        "binding_receipt_sha256",
        "binding_receipt_relative_path",
        "implementation_set_sha256",
        "official_commit",
        "qualification_file_sha256",
        "qualification_sha256",
    }:
        raise MuSiQueFormalRunnerError("official adapter binding schema drifted")
    if (
        official.get("adapter_id") != OFFICIAL_ADAPTER_ID
        or official.get("binding_receipt_relative_path")
        != OFFICIAL_BINDING_RECEIPT_RELATIVE
    ):
        raise MuSiQueFormalRunnerError("official adapter identity drifted")
    _require_sha256(
        official.get("binding_receipt_file_sha256"), "official binding receipt hash"
    )
    _require_sha256(official.get("binding_receipt_sha256"), "official binding self hash")
    for field in (
        "implementation_set_sha256",
        "qualification_file_sha256",
        "qualification_sha256",
    ):
        _require_sha256(official.get(field), field)
    if official.get("official_commit") != OFFICIAL_HIPPORAG_COMMIT:
        raise MuSiQueFormalRunnerError("official HippoRAG commit drifted")
    custody = worker.get("custody_binding")
    if not isinstance(custody, Mapping) or set(custody) != {
        "acquisition_receipt_file_sha256",
        "acquisition_sha256",
        "custody_receipt_file_sha256",
        "custody_receipt_sha256",
        "generation_view_set_sha256",
    }:
        raise MuSiQueFormalRunnerError("custody/acquisition binding schema drifted")
    for field, value in custody.items():
        _require_sha256(value, field)
    private = worker.get("private_index_binding")
    if not isinstance(private, Mapping) or set(private) != {
        "binding_sha256",
        "binding_sidecar_file_sha256",
        "custody_receipt_sha256",
        "private_index_file_sha256",
        "private_index_hash",
        "private_index_relative_path",
    }:
        raise MuSiQueFormalRunnerError("private index binding schema drifted")
    if (
        private.get("private_index_relative_path") != PRIVATE_INDEX_RELATIVE_PATH
        or private.get("custody_receipt_sha256")
        != custody.get("custody_receipt_sha256")
    ):
        raise MuSiQueFormalRunnerError("private index custody binding drifted")
    for field in (
        "binding_sha256",
        "binding_sidecar_file_sha256",
        "private_index_file_sha256",
        "private_index_hash",
    ):
        _require_sha256(private.get(field), field)
    private_path = _safe_relative_file(
        development_root, private["private_index_relative_path"], "private index"
    )
    # Pre-generation access is byte hashing only. JSON/gold parsing is delayed
    # until all eighteen generation terminals have joined.
    if _sha256_file(private_path) != private.get("private_index_file_sha256"):
        raise MuSiQueFormalRunnerError("private index exact file hash drifted")
    items = worker.get("items")
    if not isinstance(items, list) or len(items) != ITEM_COUNT:
        raise MuSiQueFormalRunnerError("worker item count drifted")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    normalized_items = []
    for ordinal, raw in enumerate(items):
        if not isinstance(raw, Mapping) or set(raw) != {
            "anonymous_item_id",
            "corpus_document_count",
            "input_relative_path",
            "input_sha256",
            "ordinal",
        }:
            raise MuSiQueFormalRunnerError("worker item schema drifted")
        anonymous = raw.get("anonymous_item_id")
        relative = raw.get("input_relative_path")
        if (
            not isinstance(anonymous, str)
            or _ANONYMOUS_ID.fullmatch(anonymous) is None
            or anonymous in seen_ids
            or raw.get("ordinal") != ordinal
            or not isinstance(relative, str)
            or relative in seen_paths
            or type(raw.get("corpus_document_count")) is not int
            or raw["corpus_document_count"] < TOP_K
        ):
            raise MuSiQueFormalRunnerError("worker item identity drifted")
        source = _safe_relative_file(development_root, relative, "gold-free item input")
        if _sha256_file(source) != _require_sha256(raw.get("input_sha256"), "input hash"):
            raise MuSiQueFormalRunnerError("gold-free item input hash drifted")
        seen_ids.add(anonymous)
        seen_paths.add(relative)
        normalized_items.append(dict(raw))
    return {**worker, "items": normalized_items}


def _load_and_verify_public_bindings(
    *,
    public_freeze_path: Path,
    custody_receipt_path: Path,
    acquisition_receipt_path: Path,
    development_root: Path,
    worker_plan_path: Path,
    worker: Mapping[str, Any],
    formal: bool,
) -> dict[str, Any]:
    """Verify every safe upstream receipt before any provider/task call."""

    from assumption_agent.benchmarks.musique_development_custody_v1 import (
        FORMAL_PUBLIC_CUSTODY_RECEIPT_RELATIVE as CUSTODY_REGISTERED_RELATIVE,
        FORMAL_PUBLIC_PRIVATE_INDEX_BINDING_RELATIVE,
        PUBLISHED_ANCHORS,
        _verify_acquisition_receipt,
        current_development_implementation_binding,
        load_public_custody_receipt,
        load_public_private_index_binding,
        verify_formal_anchor_bundle,
    )
    from assumption_agent.benchmarks.musique_development_freeze_v1 import (
        CONTROLLER_PLAN_NAME,
        FORMAL_PUBLIC_FREEZE_RELATIVE as FREEZE_REGISTERED_RELATIVE,
        verify_controller_plan,
        verify_public_pre_run_freeze,
    )

    project = _project_root()
    if formal:
        registered_freeze = project / FREEZE_REGISTERED_RELATIVE
        registered_custody = project / CUSTODY_REGISTERED_RELATIVE
        registered_acquisition = project / PUBLISHED_ANCHORS["acquisition"][
            "relative_path"
        ]
        if (
            public_freeze_path != registered_freeze
            or custody_receipt_path != registered_custody
            or acquisition_receipt_path != registered_acquisition.resolve(strict=True)
        ):
            raise MuSiQueFormalRunnerError(
                "formal execution did not receive the registered public trust roots"
            )
        anchors = verify_formal_anchor_bundle(
            preregistration_path=project
            / PUBLISHED_ANCHORS["preregistration"]["relative_path"],
            acquisition_receipt_path=registered_acquisition,
            formation_receipt_path=project
            / PUBLISHED_ANCHORS["formation"]["relative_path"],
            frozen_program_path=project
            / PUBLISHED_ANCHORS["program"]["relative_path"],
            qualification_path=project
            / PUBLISHED_ANCHORS["qualification"]["relative_path"],
            official_adapter_binding_path=project
            / PUBLISHED_ANCHORS["official_adapter"]["relative_path"],
        )
    else:
        anchors = None

    for path, field in (
        (public_freeze_path, "public freeze"),
        (custody_receipt_path, "custody receipt"),
        (acquisition_receipt_path, "acquisition receipt"),
    ):
        _reject_symlink_components(path, field)
        if not path.is_file():
            raise MuSiQueFormalRunnerError(f"{field} is not a regular file")
    try:
        freeze_raw = json.loads(public_freeze_path.read_text(encoding="utf-8"))
        freeze = verify_public_pre_run_freeze(freeze_raw)
        custody = load_public_custody_receipt(custody_receipt_path)
        acquisition, _development_binding = _verify_acquisition_receipt(
            acquisition_receipt_path
        )
    except Exception as exc:
        raise MuSiQueFormalRunnerError("upstream public receipt verification failed") from exc
    live_development_implementation = current_development_implementation_binding()
    if (
        custody.get("hashes", {}).get("development_implementation_set_sha256")
        != live_development_implementation["set_sha256"]
    ):
        raise MuSiQueFormalRunnerError(
            "custody receipt does not bind the live formal implementation"
        )
    controller_path = _safe_relative_file(
        development_root, CONTROLLER_PLAN_NAME, "controller plan"
    )
    controller = _read_hashed_json(
        controller_path, hash_field="controller_plan_hash"
    )
    try:
        verify_controller_plan(controller, worker_plan=worker)
    except Exception as exc:
        raise MuSiQueFormalRunnerError("controller plan verification failed") from exc
    if formal:
        assert anchors is not None
        anchor_payloads = anchors["payloads"]
        anchor_files = anchors["file_hashes"]
        typed = worker["typed_binding"]
        official = worker["formal_official_binding"]
        custody_hashes = custody.get("hashes", {})
        if (
            worker["custody_binding"]["acquisition_receipt_file_sha256"]
            != anchor_files["acquisition"]
            or worker["custody_binding"]["acquisition_sha256"]
            != anchor_payloads["acquisition"]["acquisition_sha256"]
            or typed["formation_receipt_file_sha256"]
            != anchor_files["formation"]
            or typed["formation_receipt_hash"]
            != anchor_payloads["formation"]["receipt_hash"]
            or typed["frozen_program_file_sha256"] != anchor_files["program"]
            or typed["frozen_program_hash"]
            != anchor_payloads["program"]["program_hash"]
            or official["binding_receipt_file_sha256"]
            != anchor_files["official_adapter"]
            or official["binding_receipt_sha256"]
            != anchor_payloads["official_adapter"]["receipt_sha256"]
            or official["qualification_file_sha256"]
            != anchor_files["qualification"]
            or official["qualification_sha256"]
            != anchor_payloads["qualification"]["qualification_sha256"]
            or custody_hashes.get("preregistration_file_sha256")
            != anchor_files["preregistration"]
            or custody_hashes.get("preregistration_sha256")
            != anchor_payloads["preregistration"]["preregistration_sha256"]
            or custody_hashes.get("formation_receipt_file_sha256")
            != anchor_files["formation"]
            or custody_hashes.get("frozen_program_file_sha256")
            != anchor_files["program"]
            or custody_hashes.get("qualification_file_sha256")
            != anchor_files["qualification"]
            or custody_hashes.get("official_adapter_binding_file_sha256")
            != anchor_files["official_adapter"]
        ):
            raise MuSiQueFormalRunnerError(
                "formal worker/custody chain differs from the published trust roots"
            )
        private_sidecar_path = (
            project / FORMAL_PUBLIC_PRIVATE_INDEX_BINDING_RELATIVE
        ).resolve(strict=True)
        private_sidecar = load_public_private_index_binding(private_sidecar_path)
        if (
            _sha256_file(private_sidecar_path)
            != worker["private_index_binding"]["binding_sidecar_file_sha256"]
            or private_sidecar.get("binding_sha256")
            != worker["private_index_binding"]["binding_sha256"]
        ):
            raise MuSiQueFormalRunnerError(
                "formal private-index sidecar differs from its registered trust root"
            )
    authorization = freeze.get("authorization")
    bindings = freeze.get("binding_hashes")
    if not isinstance(authorization, Mapping) or not isinstance(bindings, Mapping):
        raise MuSiQueFormalRunnerError("public freeze bindings are malformed")
    expected_authorization = {
        "authorization_id": worker["authorization_id"],
        "consumption_marker_relative_path": CONSUMPTION_MARKER_RELATIVE_PATH,
        "development_root_commitment": worker["development_root_commitment"],
        "execution_root_commitment": worker["execution_root_commitment"],
        "execution_root_relative_path": EXECUTION_ROOT_RELATIVE_PATH,
        "launch_authorized": True,
    }
    if dict(authorization) != expected_authorization:
        raise MuSiQueFormalRunnerError("public freeze authorization drifted")
    custody_binding = worker["custody_binding"]
    if (
        _sha256_file(custody_receipt_path)
        != custody_binding["custody_receipt_file_sha256"]
        or custody.get("receipt_sha256")
        != custody_binding["custody_receipt_sha256"]
        or _sha256_file(acquisition_receipt_path)
        != custody_binding["acquisition_receipt_file_sha256"]
        or acquisition.get("acquisition_sha256")
        != custody_binding["acquisition_sha256"]
    ):
        raise MuSiQueFormalRunnerError("custody/acquisition exact binding drifted")
    if (
        custody.get("hashes", {}).get("development_file_sha256")
        != _development_binding.get("file_sha256")
        or custody.get("hashes", {}).get(
            "development_item_commitment_set_sha256"
        )
        != _development_binding.get("item_commitment_set_sha256")
        or custody.get("hashes", {}).get("private_pack_sha256")
        != acquisition.get("commitments", {}).get("private_pack_sha256")
    ):
        raise MuSiQueFormalRunnerError(
            "custody receipt is not bound to the published development split"
        )
    expected_cross_bindings = {
        "acquisition_receipt_file_sha256": custody_binding[
            "acquisition_receipt_file_sha256"
        ],
        "acquisition_sha256": custody_binding["acquisition_sha256"],
        "custody_receipt_file_sha256": custody_binding[
            "custody_receipt_file_sha256"
        ],
        "custody_receipt_sha256": custody_binding["custody_receipt_sha256"],
        "formation_receipt_file_sha256": worker["typed_binding"][
            "formation_receipt_file_sha256"
        ],
        "formation_receipt_hash": worker["typed_binding"][
            "formation_receipt_hash"
        ],
        "frozen_program_file_sha256": worker["typed_binding"][
            "frozen_program_file_sha256"
        ],
        "frozen_program_hash": worker["typed_binding"]["frozen_program_hash"],
        "official_adapter_binding_file_sha256": worker[
            "formal_official_binding"
        ]["binding_receipt_file_sha256"],
        "official_adapter_binding_receipt_sha256": worker["formal_official_binding"][
            "binding_receipt_sha256"
        ],
        "official_adapter_implementation_set_sha256": worker[
            "formal_official_binding"
        ]["implementation_set_sha256"],
        "qualification_file_sha256": worker["formal_official_binding"][
            "qualification_file_sha256"
        ],
        "qualification_sha256": worker["formal_official_binding"][
            "qualification_sha256"
        ],
        "private_index_binding_sidecar_file_sha256": worker[
            "private_index_binding"
        ]["binding_sidecar_file_sha256"],
        "private_index_binding_sha256": worker["private_index_binding"][
            "binding_sha256"
        ],
        "private_index_file_sha256": worker["private_index_binding"][
            "private_index_file_sha256"
        ],
        "private_index_hash": worker["private_index_binding"][
            "private_index_hash"
        ],
        "provider_identity_sha256": worker["provider_identity"][
            "identity_sha256"
        ],
        "runner_implementation_set_sha256": worker["runner_implementation"][
            "set_sha256"
        ],
        "development_implementation_set_sha256": live_development_implementation[
            "set_sha256"
        ],
        "controller_plan_file_sha256": _sha256_file(controller_path),
        "controller_plan_hash": controller["controller_plan_hash"],
        "shared_contract_sha256": stable_hash(worker["shared_contract"]),
        "worker_plan_file_sha256": _sha256_file(worker_plan_path),
        "worker_plan_hash": worker["worker_plan_hash"],
    }
    for field, expected in expected_cross_bindings.items():
        if bindings.get(field) != expected:
            raise MuSiQueFormalRunnerError(f"public freeze cross-binding drifted: {field}")
    if formal:
        assert anchors is not None
        if (
            bindings.get("preregistration_file_sha256")
            != anchors["file_hashes"]["preregistration"]
            or bindings.get("preregistration_sha256")
            != anchors["payloads"]["preregistration"]["preregistration_sha256"]
        ):
            raise MuSiQueFormalRunnerError(
                "public freeze preregistration trust root drifted"
            )
    if freeze.get("protocol_amendment", {}).get(
        "formal_budget_bytes"
    ) != MODEL_REQUEST_BODY_BYTE_BUDGET:
        raise MuSiQueFormalRunnerError("public freeze byte-budget amendment drifted")
    return freeze


def _verify_official_binding_receipt(worker: Mapping[str, Any]) -> Path:
    official = worker["formal_official_binding"]
    receipt = _project_root() / OFFICIAL_BINDING_RECEIPT_RELATIVE
    _reject_symlink_components(receipt, "official adapter binding receipt")
    if not receipt.is_file() or _sha256_file(receipt) != official[
        "binding_receipt_file_sha256"
    ]:
        raise MuSiQueFormalRunnerError("official adapter binding file drifted")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MuSiQueFormalRunnerError("official adapter binding is unreadable") from exc
    declared = payload.get("receipt_sha256") if isinstance(payload, Mapping) else None
    body = dict(payload) if isinstance(payload, Mapping) else {}
    body.pop("receipt_sha256", None)
    implementation = payload.get("implementation_binding") if isinstance(payload, Mapping) else None
    qualification = payload.get("qualification_binding") if isinstance(payload, Mapping) else None
    source = payload.get("official_source_binding") if isinstance(payload, Mapping) else None
    if (
        declared != official["binding_receipt_sha256"]
        or stable_hash(body) != declared
        or not isinstance(implementation, Mapping)
        or implementation.get("set_sha256") != official["implementation_set_sha256"]
        or not isinstance(qualification, Mapping)
        or qualification.get("file_sha256") != official["qualification_file_sha256"]
        or qualification.get("qualification_sha256")
        != official["qualification_sha256"]
        or not isinstance(source, Mapping)
        or source.get("commit") != official["official_commit"]
    ):
        raise MuSiQueFormalRunnerError("official adapter receipt closure drifted")
    return receipt


def _load_gold_free_item(path: Path, expected: Mapping[str, Any]) -> tuple[str, tuple[dict[str, object], ...]]:
    try:
        raw_bytes = path.read_bytes()
        if _sha256_bytes(raw_bytes) != expected.get("input_sha256"):
            raise MuSiQueFormalRunnerError(
                "gold-free item changed at its actual use point"
            )
        value = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MuSiQueFormalRunnerError("gold-free item is unreadable") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "anonymous_item_id", "corpus", "question", "schema"
    }:
        raise MuSiQueFormalRunnerError("gold-free item schema drifted")
    if (
        value.get("schema") != ITEM_INPUT_VERSION
        or value.get("anonymous_item_id") != expected["anonymous_item_id"]
    ):
        raise MuSiQueFormalRunnerError("gold-free item identity drifted")
    question = value.get("question")
    corpus = value.get("corpus")
    if not isinstance(question, str) or not question.strip() or not isinstance(corpus, list):
        raise MuSiQueFormalRunnerError("gold-free question/corpus is malformed")
    if len(corpus) != expected["corpus_document_count"] or len(corpus) < TOP_K:
        raise MuSiQueFormalRunnerError("gold-free corpus count drifted")
    paragraphs: list[dict[str, object]] = []
    for idx, raw in enumerate(corpus):
        if not isinstance(raw, Mapping) or set(raw) != {"idx", "paragraph_text", "title"}:
            raise MuSiQueFormalRunnerError("gold-free paragraph schema drifted")
        if (
            raw.get("idx") != idx
            or not isinstance(raw.get("title"), str)
            or not raw["title"].strip()
            or not isinstance(raw.get("paragraph_text"), str)
            or not raw["paragraph_text"].strip()
        ):
            raise MuSiQueFormalRunnerError("gold-free paragraph identity drifted")
        paragraphs.append(dict(raw))
    serialized = _canonical_bytes(value)
    for forbidden in (b'"answer"', b'"answers"', b'"support_indices"', b'"is_supporting"'):
        if forbidden in serialized:
            raise MuSiQueFormalRunnerError("gold material entered worker input")
    return question.strip(), tuple(paragraphs)


def _validate_retrieved_indices(indices: Sequence[int], *, document_count: int) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise MuSiQueFormalRunnerError("retriever output is not an index sequence")
    normalized = tuple(indices)
    if (
        len(normalized) != TOP_K
        or any(type(index) is not int for index in normalized)
        or len(set(normalized)) != TOP_K
        or any(index < 0 or index >= document_count for index in normalized)
    ):
        raise MuSiQueFormalRunnerError("retriever did not return exact in-corpus top-5")
    return normalized


def _serialize_context(paragraphs: Sequence[Mapping[str, object]], indices: Sequence[int]) -> str:
    by_idx = {int(row["idx"]): row for row in paragraphs}
    blocks = []
    for rank, index in enumerate(indices, start=1):
        row = by_idx[index]
        blocks.append(
            "\n".join(
                (
                    f"[DOCUMENT {rank}]",
                    f"Title: {row['title']}",
                    f"Text: {row['paragraph_text']}",
                )
            )
        )
    return "\n\n".join(blocks)


def _generator_request(*, item_id: str, arm: str, question: str, context: str) -> ModelRequest:
    request = ModelRequest(
        purpose=f"musique_generator:{item_id}:{arm}",
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_payload={
            "answer_contract": {
                "additionalProperties": False,
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "type": "object",
            },
            "context": context,
            "context_serialization_version": CONTEXT_SERIALIZATION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "question": question,
        },
        max_output_tokens=MODEL_OUTPUT_TOKEN_BUDGET,
        json_object=True,
    )
    if request.canonical_body_byte_count > MODEL_REQUEST_BODY_BYTE_BUDGET:
        raise MuSiQueFormalRunnerError("complete generator request exceeds 64 KiB")
    if request.body().get("model") != MODEL_ID or request.body().get("temperature") != 0:
        raise MuSiQueFormalRunnerError("generator model contract drifted")
    return request


@dataclass(frozen=True)
class PreparedWork:
    anonymous_item_id: str
    arm: str
    retrieved_indices: tuple[int, ...]
    request: ModelRequest

    @property
    def work_unit_id(self) -> str:
        return f"{self.anonymous_item_id}:{self.arm}"

    @property
    def work_unit_hash(self) -> str:
        return stable_hash(
            {
                "arm": self.arm,
                "request_hash": self.request.request_hash,
                "retrieved_index_hash": stable_hash(list(self.retrieved_indices)),
                "work_unit_id": self.work_unit_id,
            }
        )


def _prepare_item(
    *,
    development_root: Path,
    item: Mapping[str, Any],
    typed_binding: Mapping[str, Any],
    typed_program_path: Path,
    typed_receipt_path: Path,
    official_retrieve: OfficialRetrieveOnly,
    official_work_root: Path,
) -> tuple[PreparedWork, ...]:
    source = _safe_relative_file(development_root, item["input_relative_path"], "gold-free item input")
    question, paragraphs = _load_gold_free_item(source, item)
    if (
        _sha256_file(typed_receipt_path)
        != typed_binding.get("formation_receipt_file_sha256")
        or _sha256_file(typed_program_path)
        != typed_binding.get("frozen_program_file_sha256")
    ):
        raise MuSiQueFormalRunnerError(
            "typed retriever changed at its actual use point"
        )
    receipt = load_formation_receipt(typed_receipt_path)
    program = load_frozen_program(
        typed_program_path,
        receipt_path=typed_receipt_path,
        verify_live=True,
    )
    if (
        receipt.get("receipt_hash")
        != typed_binding.get("formation_receipt_hash")
        or program.program_hash != typed_binding.get("frozen_program_hash")
    ):
        raise MuSiQueFormalRunnerError(
            "typed retriever semantic binding changed at its actual use point"
        )
    typed_corpus = tuple(
        RetrievalParagraph(
            idx=int(row["idx"]),
            title=str(row["title"]),
            text=str(row["paragraph_text"]),
        )
        for row in paragraphs
    )
    by_arm = {
        ARM_IDS[0]: tuple(range(TOP_K)),
        ARM_IDS[1]: typed_retrieve(program, question, typed_corpus),
        ARM_IDS[2]: official_retrieve(
            question=question,
            paragraphs=paragraphs,
            work_root=official_work_root,
        ),
    }
    prepared = []
    for arm in ARM_IDS:
        indices = _validate_retrieved_indices(
            by_arm[arm], document_count=len(paragraphs)
        )
        context = _serialize_context(paragraphs, indices)
        request = _generator_request(
            item_id=str(item["anonymous_item_id"]),
            arm=arm,
            question=question,
            context=context,
        )
        prepared.append(
            PreparedWork(
                anonymous_item_id=str(item["anonymous_item_id"]),
                arm=arm,
                retrieved_indices=indices,
                request=request,
            )
        )
    body_shapes = {
        stable_hash(
            {
                "answer_contract": request.request.user_payload["answer_contract"],
                "context_serialization_version": request.request.user_payload[
                    "context_serialization_version"
                ],
                "prompt_version": request.request.user_payload["prompt_version"],
                "system_prompt": request.request.system_prompt,
            }
        )
        for request in prepared
    }
    if len(body_shapes) != 1:
        raise MuSiQueFormalRunnerError("three arms do not share one generator template")
    return tuple(prepared)


def _strict_answer(content: str) -> str:
    stripped = content.strip()
    try:
        value = json.loads(
            stripped,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MuSiQueOutputContractError("generator response is not strict JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"answer"}
        or not isinstance(value.get("answer"), str)
        or not value["answer"].strip()
        or _canonical_bytes(value).decode("utf-8") != stripped
    ):
        raise MuSiQueOutputContractError("generator answer contract is invalid")
    return value["answer"].strip()


class _ConcurrencyCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.maximum = max(self.maximum, self._active)

    def leave(self) -> None:
        with self._lock:
            self._active -= 1


def _write_provider_precommit(
    *, output_root: Path, plus: ProviderCredential, pro: ProviderCredential, formal: bool
) -> dict[str, Any]:
    if (
        plus.api_origin != PROVIDER_ORIGIN
        or pro.api_origin != PROVIDER_ORIGIN
        or plus.model != MODEL_ID
        or pro.model != MODEL_ID
        or plus.api_key_hmac_sha256 == pro.api_key_hmac_sha256
    ):
        raise MuSiQueFormalRunnerError("provider precommit identity drifted")
    body = {
        "canary_contains_development_content": False,
        "dependency_injection_used": not formal,
        "formal_evidence": formal,
        "model": MODEL_ID,
        "precommit_version": PROVIDER_PRECOMMIT_VERSION,
        "providers": {
            "plus": plus.safe_identity(provider_label="plus"),
            "pro": pro.safe_identity(provider_label="pro"),
        },
        "secret_value_persisted": False,
        "task_calls_before_precommit": 0,
    }
    return atomic_write_hashed_json_v2(
        output_root / PROVIDER_PRECOMMIT_FILENAME, body, hash_field="precommit_hash"
    )


def _canary(
    *,
    output_root: Path,
    label: str,
    credential: ProviderCredential,
    transport: ModelTransport,
    precommit_hash: str,
) -> dict[str, Any]:
    request = ModelRequest(
        purpose="provider_transport_canary",
        system_prompt=CANARY_SYSTEM_PROMPT,
        user_payload=CANARY_PAYLOAD,
        max_output_tokens=32,
        json_object=False,
    )
    claim = atomic_write_hashed_json_v2(
        output_root / "provider_canary" / f"{label}.claim.json",
        {
            "attempt_count": 1,
            "development_content_accessed": False,
            "precommit_hash": precommit_hash,
            "provider_label": label,
            "replay_authorized": False,
            "request_hash": request.request_hash,
        },
        hash_field="claim_hash",
    )
    try:
        content = transport.complete(credential=credential, request=request)
        if not isinstance(content, str) or not content.strip():
            raise ProviderProtocolError("canary response is empty")
    except ProviderTransportUnavailable:
        return atomic_write_hashed_json_v2(
            output_root / "provider_canary" / f"{label}.failure.json",
            {
                "claim_hash": claim["claim_hash"],
                "failure_kind": "transport_unavailable",
                "model_call_count": 1,
                "model_response_received": False,
                "raw_response_persisted": False,
            },
            hash_field="receipt_hash",
        )
    except Exception as exc:
        atomic_write_hashed_json_v2(
            output_root / "provider_canary" / f"{label}.failure.json",
            {
                "claim_hash": claim["claim_hash"],
                "failure_kind": "provider_protocol_failure",
                "model_call_count": 1,
                "model_response_received": False,
                "raw_response_persisted": False,
            },
            hash_field="receipt_hash",
        )
        raise ProviderProtocolError("provider canary failed without authorized fallback") from exc
    return atomic_write_hashed_json_v2(
        output_root / "provider_canary" / f"{label}.success.json",
        {
            "claim_hash": claim["claim_hash"],
            "failure_kind": None,
            "model_call_count": 1,
            "model_response_received": True,
            "raw_response_persisted": False,
            "response_hash": stable_hash({"content": content}),
        },
        hash_field="receipt_hash",
    )


def _select_provider(
    *,
    output_root: Path,
    plus: ProviderCredential,
    pro: ProviderCredential,
    transport: ModelTransport,
    precommit: Mapping[str, Any],
) -> tuple[str, ProviderCredential, dict[str, Any], int]:
    plus_outcome = _canary(
        output_root=output_root,
        label="plus",
        credential=plus,
        transport=transport,
        precommit_hash=str(precommit["precommit_hash"]),
    )
    if plus_outcome.get("model_response_received") is True:
        label, credential, pro_hash, calls = "plus", plus, None, 1
        probe_order = ["plus_complete_model_response"]
    elif plus_outcome.get("failure_kind") == "transport_unavailable":
        pro_outcome = _canary(
            output_root=output_root,
            label="pro",
            credential=pro,
            transport=transport,
            precommit_hash=str(precommit["precommit_hash"]),
        )
        if pro_outcome.get("model_response_received") is not True:
            raise ProviderTransportUnavailable("Pro canary is unavailable")
        label, credential, pro_hash, calls = "pro", pro, pro_outcome["receipt_hash"], 2
        probe_order = ["plus_transport_unavailable", "pro_complete_model_response"]
    else:
        raise ProviderProtocolError("Plus did not satisfy the frozen fallback condition")
    selection = atomic_write_hashed_json_v2(
        output_root / PROVIDER_SELECTION_FILENAME,
        {
            "mid_batch_provider_switch_authorized": False,
            "precommit_hash": precommit["precommit_hash"],
            "probe_order": probe_order,
            "pro_canary_receipt_hash": pro_hash,
            "plus_canary_receipt_hash": plus_outcome["receipt_hash"],
            "replay_authorized": False,
            "resampling_authorized": False,
            "retry_authorized": False,
            "selected_provider_fixed_for_complete_batch": True,
            "selected_provider_label": label,
            "selection_completed_before_task_calls": True,
            "selection_version": PROVIDER_SELECTION_VERSION,
            "semantic_acceptance_used_for_selection": False,
            "task_calls_before_selection": 0,
        },
        hash_field="selection_hash",
    )
    return label, credential, selection, calls


def _preclaim(
    *, output_root: Path, prepared: Sequence[PreparedWork], selection_hash: str
) -> dict[str, dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    for work in prepared:
        claim = atomic_write_hashed_json_v2(
            output_root / "generation_state" / work.work_unit_hash / "claim.json",
            {
                "attempt_count": 1,
                "claim_version": CLAIM_VERSION,
                "model_replay_authorized": False,
                "request_hash": work.request.request_hash,
                "resampling_authorized": False,
                "retry_authorized": False,
                "selection_hash": selection_hash,
                "work_unit_hash": work.work_unit_hash,
                "work_unit_id": work.work_unit_id,
            },
            hash_field="claim_hash",
        )
        claims[work.work_unit_id] = claim
    if len(claims) != WORK_UNIT_COUNT:
        raise MuSiQueNoReplayError("all 18 claims were not durably persisted")
    atomic_write_hashed_json_v2(
        output_root / CLAIM_SET_FILENAME,
        {
            "all_claims_persisted_before_work_start": True,
            "claim_count": WORK_UNIT_COUNT,
            "claim_hash_set_sha256": stable_hash(
                sorted(claim["claim_hash"] for claim in claims.values())
            ),
            "claim_set_version": CLAIM_SET_VERSION,
            "selection_hash": selection_hash,
        },
        hash_field="claim_set_hash",
    )
    return claims


def _run_generator_once(
    *,
    output_root: Path,
    work: PreparedWork,
    claim: Mapping[str, Any],
    selected_label: str,
    credential: ProviderCredential,
    transport: ModelTransport,
    start_barrier: threading.Barrier,
    concurrency: _ConcurrencyCounter,
) -> dict[str, Any]:
    start_barrier.wait(timeout=60)
    prediction: str | None = None
    error_code: str | None = None
    response_hash: str | None = None
    response_received = False
    concurrency.enter()
    try:
        try:
            content = transport.complete(credential=credential, request=work.request)
            response_received = True
            response_hash = stable_hash({"content": content})
            prediction = _strict_answer(content)
        except ProviderTransportUnavailable:
            error_code = "selected_provider_transport_unavailable"
        except ProviderProtocolError:
            error_code = "selected_provider_protocol_failure"
        except MuSiQueOutputContractError:
            error_code = "generator_output_contract_invalid"
        except Exception:
            error_code = "generator_execution_failed"
    finally:
        concurrency.leave()
    return atomic_write_hashed_json_v2(
        output_root / "generation_state" / work.work_unit_hash / "terminal.private.json",
        {
            "arm": work.arm,
            "attempt_count": 1,
            "claim_hash": claim["claim_hash"],
            "error_code": error_code,
            "execution_terminal": True,
            "model_call_count": 1,
            "model_response_hash": response_hash,
            "model_response_received": response_received,
            "output_contract_valid": prediction is not None,
            "prediction": prediction,
            "prediction_hash": stable_hash({"prediction": prediction}) if prediction is not None else None,
            "prompt_persisted": False,
            "provider_label_hash": stable_hash({"provider_label": selected_label}),
            "raw_response_persisted": False,
            "secret_value_persisted": False,
            "terminal_version": TERMINAL_VERSION,
            "trace_persisted": False,
            "work_unit_hash": work.work_unit_hash,
            "work_unit_id": work.work_unit_id,
        },
        hash_field="terminal_hash",
    )


def _load_private_index(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_index_hash: str,
    expected_custody_receipt_sha256: str,
    prepared: Sequence[PreparedWork],
) -> dict[str, Mapping[str, Any]]:
    if _sha256_file(path) != expected_file_sha256:
        raise MuSiQueFormalRunnerError("private index changed after pre-generation binding")
    private = _read_hashed_json(path, hash_field="private_index_hash")
    if set(private) != {
        "custody_receipt_sha256",
        "items",
        "private_index_hash",
        "private_index_version",
    }:
        raise MuSiQueFormalRunnerError("private index schema drifted")
    if (
        private.get("private_index_version") != PRIVATE_INDEX_VERSION
        or private.get("private_index_hash") != expected_index_hash
        or private.get("custody_receipt_sha256")
        != expected_custody_receipt_sha256
        or not isinstance(private.get("items"), list)
        or len(private["items"]) != ITEM_COUNT
    ):
        raise MuSiQueFormalRunnerError("private index identity drifted")
    expected_ids = {work.anonymous_item_id for work in prepared}
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in private["items"]:
        if not isinstance(row, Mapping) or set(row) != {
            "accepted_aliases", "anonymous_item_id", "support_indices"
        }:
            raise MuSiQueFormalRunnerError("private index item schema drifted")
        anonymous = row.get("anonymous_item_id")
        aliases = row.get("accepted_aliases")
        support = row.get("support_indices")
        if (
            anonymous not in expected_ids
            or anonymous in by_id
            or not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(value, str) or not value.strip() for value in aliases)
            or not isinstance(support, list)
            or not support
            or any(type(index) is not int or index < 0 for index in support)
            or len(set(support)) != len(support)
        ):
            raise MuSiQueFormalRunnerError("private index item content drifted")
        by_id[str(anonymous)] = row
    if set(by_id) != expected_ids:
        raise MuSiQueFormalRunnerError("private index item set drifted")
    return by_id


def _offline_evaluate(
    *,
    output_root: Path,
    private_index_path: Path,
    private_index_binding: Mapping[str, Any],
    prepared: Sequence[PreparedWork],
    terminals: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(terminals) != WORK_UNIT_COUNT or any(
        terminal.get("execution_terminal") is not True for terminal in terminals.values()
    ):
        raise MuSiQueFormalRunnerError("private index cannot release before 18 terminal joins")
    private = _load_private_index(
        private_index_path,
        expected_file_sha256=str(
            private_index_binding["private_index_file_sha256"]
        ),
        expected_index_hash=str(private_index_binding["private_index_hash"]),
        expected_custody_receipt_sha256=str(
            private_index_binding["custody_receipt_sha256"]
        ),
        prepared=prepared,
    )
    rows = []
    oracle_disagreements = 0
    answer_oracle_calls = 0
    support_oracle_calls = 0
    for work in prepared:
        gold = private[work.anonymous_item_id]
        terminal = terminals[work.work_unit_id]
        output_valid = terminal.get("output_contract_valid") is True
        prediction = terminal.get("prediction") if output_valid else None
        if output_valid and not isinstance(prediction, str):
            raise MuSiQueFormalRunnerError("valid terminal lacks a text prediction")
        if output_valid:
            primary_answer = evaluate_aliases_primary(
                str(prediction), gold["accepted_aliases"]
            )
            secondary_answer = evaluate_aliases_secondary(
                str(prediction), gold["accepted_aliases"]
            )
            answer_oracle_calls += 2
        else:
            # ITT invalid/transport outcomes are assigned exact zero directly;
            # an empty synthetic prediction is never passed to a normalizer.
            primary_answer = secondary_answer = (0, Fraction(0, 1))
        primary_support = evaluate_support_primary(work.retrieved_indices, gold["support_indices"])
        secondary_support = evaluate_support_secondary(work.retrieved_indices, gold["support_indices"])
        support_oracle_calls += 2
        disagreement = primary_answer != secondary_answer or primary_support != secondary_support
        oracle_disagreements += int(disagreement)
        if disagreement:
            raise MuSiQueFormalRunnerError("independent offline MuSiQue oracles disagree")
        rows.append(
            {
                "anonymous_item_id": work.anonymous_item_id,
                "answer_exact": primary_answer[0],
                "answer_f1": _fraction_payload(primary_answer[1]),
                "arm": work.arm,
                "output_contract_valid": output_valid,
                "prediction_hash": terminal.get("prediction_hash"),
                "support_recall_at_5": _fraction_payload(primary_support),
                "terminal_hash": terminal["terminal_hash"],
            }
        )
    private_receipt = atomic_write_hashed_json_v2(
        output_root / PRIVATE_EVALUATION_FILENAME,
        {
            "evaluation_version": PRIVATE_EVALUATION_VERSION,
            "answer_oracle_call_count": answer_oracle_calls,
            "item_arm_row_count": len(rows),
            "oracle_disagreement_count": oracle_disagreements,
            "oracle_release_after_terminal_join_count": WORK_UNIT_COUNT,
            "rows": rows,
            "support_oracle_call_count": support_oracle_calls,
        },
        hash_field="evaluation_hash",
    )
    metrics = ("answer_exact", "answer_f1", "support_recall_at_5")
    arm_values: dict[str, dict[str, list[Fraction]]] = {
        arm: {metric: [] for metric in metrics} for arm in ARM_IDS
    }
    valid_counts = {arm: 0 for arm in ARM_IDS}
    for row in rows:
        arm = str(row["arm"])
        valid_counts[arm] += int(row["output_contract_valid"])
        arm_values[arm]["answer_exact"].append(Fraction(int(row["answer_exact"]), 1))
        arm_values[arm]["answer_f1"].append(
            Fraction(row["answer_f1"]["numerator"], row["answer_f1"]["denominator"])
        )
        arm_values[arm]["support_recall_at_5"].append(
            Fraction(
                row["support_recall_at_5"]["numerator"],
                row["support_recall_at_5"]["denominator"],
            )
        )
    aggregates: dict[str, Any] = {}
    for arm in ARM_IDS:
        aggregates[arm] = {"generator_output_contract_valid_count": valid_counts[arm]}
        for metric in metrics:
            values = arm_values[arm][metric]
            total = sum(values, Fraction(0, 1))
            aggregates[arm][metric] = {
                "mean": _fraction_payload(total / ITEM_COUNT),
                "sum": _fraction_payload(total),
            }
    contrast_specs = {
        "assumption_minus_canonical": (ARM_IDS[1], ARM_IDS[0]),
        "official_hipporag_minus_canonical": (ARM_IDS[2], ARM_IDS[0]),
        "assumption_minus_official_hipporag": (ARM_IDS[1], ARM_IDS[2]),
    }
    pairwise: dict[str, Any] = {}
    for name, (first, second) in contrast_specs.items():
        pairwise[name] = {}
        for metric in metrics:
            deltas = [
                left - right
                for left, right in zip(arm_values[first][metric], arm_values[second][metric])
            ]
            pairwise[name][metric] = {
                "gain_count": sum(delta > 0 for delta in deltas),
                "harm_count": sum(delta < 0 for delta in deltas),
                "tie_count": sum(delta == 0 for delta in deltas),
                "paired_count": len(deltas),
                "paired_delta_sum": _fraction_payload(sum(deltas, Fraction(0, 1))),
            }
    return {
        "answer_oracle_call_count": answer_oracle_calls,
        "arm_aggregates": aggregates,
        "intention_to_treat_invalid_generator_output_as_incorrect": True,
        "oracle_disagreement_count": oracle_disagreements,
        "pairwise_item_counts": pairwise,
        "support_oracle_call_count": support_oracle_calls,
    }, private_receipt


def _assert_public_aggregate_only(report: Mapping[str, Any], credentials: Sequence[ProviderCredential]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                    raise MuSiQueFormalRunnerError("public report contains item content")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(report)
    raw = _canonical_bytes(report)
    if any(credential.api_key.encode("utf-8") in raw for credential in credentials):
        raise MuSiQueFormalRunnerError("public report contains provider secret")


def _formal_adapter(
    *, runtime_python: Path, local_llm_model: Path, local_embedding_model: Path
) -> OfficialRetrieveOnly:
    module = importlib.import_module(OFFICIAL_ADAPTER_MODULE)
    function = getattr(module, OFFICIAL_ADAPTER_FUNCTION, None)
    if not callable(function):
        raise MuSiQueFormalRunnerError("fixed official retrieve-only adapter is unavailable")
    expected_source = (
        _project_root()
        / "replication_runtime/musique_official_hipporag_v1/adapter.py"
    ).resolve(strict=True)
    observed_source_raw = inspect.getsourcefile(function)
    if (
        function.__module__ != OFFICIAL_ADAPTER_MODULE
        or observed_source_raw is None
        or Path(observed_source_raw).resolve(strict=True) != expected_source
    ):
        raise MuSiQueFormalRunnerError("official retrieve-only adapter source drifted")
    signature = inspect.signature(function)
    if set(signature.parameters) != {
        "question",
        "paragraphs",
        "runtime_python",
        "local_llm_model",
        "local_embedding_model",
        "binding_receipt_path",
        "work_root",
        "timeout_seconds",
    }:
        raise MuSiQueFormalRunnerError("official retrieve-only adapter interface drifted")
    binding_receipt = _project_root() / OFFICIAL_BINDING_RECEIPT_RELATIVE

    def retrieve(
        *, question: str, paragraphs: Sequence[Mapping[str, object]], work_root: Path
    ) -> tuple[int, ...]:
        result = function(
            question=question,
            paragraphs=paragraphs,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            binding_receipt_path=binding_receipt,
            work_root=work_root,
            timeout_seconds=900,
        )
        return tuple(result)

    return retrieve


@dataclass(frozen=True)
class _FormalRuntimePaths:
    runtime_python: Path
    local_llm_model: Path
    local_embedding_model: Path


def _consume_global_authorization(
    *,
    development_root: Path,
    worker: Mapping[str, Any],
    public_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    marker = development_root / CONSUMPTION_MARKER_RELATIVE_PATH
    _reject_symlink_components(marker, "authorization consumption marker")
    failure_marker = development_root / (
        f"execution.failure.{worker['authorization_id']}.json"
    )
    if (
        marker.exists()
        or marker.is_symlink()
        or failure_marker.exists()
        or failure_marker.is_symlink()
    ):
        raise MuSiQueNoReplayError(
            "study authorization was already consumed; changing output root cannot replay it"
        )
    return atomic_write_hashed_json_v2(
        marker,
        {
            "authorization_id": worker["authorization_id"],
            "consumption_marker_version": "musique_execution_authorization_consumption_v1",
            "execution_root_commitment": worker["execution_root_commitment"],
            "formal_replay_authorized": False,
            "public_freeze_hash": public_freeze["freeze_sha256"],
            "study_id": worker["study_id"],
            "worker_plan_hash": worker["worker_plan_hash"],
        },
        hash_field="consumption_hash",
    )


def _write_global_failure_receipt(
    *,
    development_root: Path,
    worker: Mapping[str, Any] | None,
    exc: BaseException,
    authorization_consumed: bool,
    external_ruoli_calls_started: bool,
    generation_work_may_have_started: bool,
    gold_open_may_have_started: bool,
    private_evaluation_may_have_started: bool,
) -> None:
    authorization = (
        str(worker.get("authorization_id"))
        if isinstance(worker, Mapping)
        else stable_hash({"unbound_preflight_failure": True})
    )
    body = {
        "authorization_consumed": authorization_consumed,
        "authorization_id": authorization,
        "error_message_sha256": stable_hash({"message": str(exc)}),
        "error_type": type(exc).__name__,
        "external_ruoli_calls_started": external_ruoli_calls_started,
        "failure_receipt_version": "musique_three_arm_safe_failure_v1",
        "gold_open_may_have_started": gold_open_may_have_started,
        "private_evaluation_may_have_started": private_evaluation_may_have_started,
        "private_prediction_or_terminal_may_be_persisted": (
            generation_work_may_have_started
        ),
        "public_aggregate_report_persisted": (
            development_root
            / EXECUTION_ROOT_RELATIVE_PATH
            / PUBLIC_REPORT_FILENAME
        ).is_file(),
        "public_content_leak_detected": False,
        "replay_authorized": False,
        "secret_persisted": False,
        "trace_persisted": False,
    }
    try:
        atomic_write_hashed_json_v2(
            development_root / f"execution.failure.{authorization}.json",
            body,
            hash_field="failure_hash",
        )
    except Exception:
        # Failure reporting must never mask the original exception. The global
        # consumption marker, once present, remains the replay authority.
        pass


def _run_core(
    *,
    development_root: str | Path,
    public_freeze_path: str | Path,
    custody_receipt_path: str | Path,
    acquisition_receipt_path: str | Path,
    output_root: str | Path,
    plus_env_file: str | Path,
    pro_env_file: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
    formal_runtime: _FormalRuntimePaths | None = None,
    synthetic_transport: ModelTransport | None = None,
    synthetic_official_retrieve: OfficialRetrieveOnly | None = None,
) -> dict[str, Any]:
    formal = formal_runtime is not None
    if formal:
        if synthetic_transport is not None or synthetic_official_retrieve is not None:
            raise MuSiQueFormalRunnerError(
                "formal execution cannot accept injected dependencies"
            )
    elif synthetic_transport is None or synthetic_official_retrieve is None:
        raise MuSiQueFormalRunnerError(
            "synthetic execution requires explicitly injected non-formal dependencies"
        )

    development_lexical = _reject_symlink_components(
        development_root, "development root"
    )
    if not development_lexical.is_dir():
        raise MuSiQueFormalRunnerError("development root is not a regular directory")
    development = development_lexical.resolve(strict=True)
    destination = _reject_symlink_components(output_root, "execution root")
    expected_destination = development / EXECUTION_ROOT_RELATIVE_PATH
    if destination != expected_destination:
        raise MuSiQueFormalRunnerError(
            "execution root differs from the uniquely authorized fixed root"
        )
    if destination.exists() or destination.is_symlink():
        raise MuSiQueNoReplayError(
            "fixed execution root was already consumed; replay is forbidden"
        )
    _require_ignored_or_external(destination)
    worker_plan_path = _safe_relative_file(
        development, WORKER_PLAN_FILENAME, "worker plan"
    )
    worker: dict[str, Any] | None = None
    authorization_consumed = False
    external_ruoli_calls_started = False
    generation_work_may_have_started = False
    gold_open_may_have_started = False
    private_evaluation_may_have_started = False
    plus: ProviderCredential | None = None
    pro: ProviderCredential | None = None
    try:
        worker = _validate_worker_plan(
            worker_plan_path,
            development_root=development,
            plus_channel_id=plus_channel_id,
            pro_channel_id=pro_channel_id,
        )
        public_freeze_source = _reject_symlink_components(
            public_freeze_path, "public freeze"
        ).resolve(strict=True)
        custody_source = _reject_symlink_components(
            custody_receipt_path, "custody receipt"
        ).resolve(strict=True)
        acquisition_source = _reject_symlink_components(
            acquisition_receipt_path, "acquisition receipt"
        ).resolve(strict=True)
        public_freeze = _load_and_verify_public_bindings(
            public_freeze_path=public_freeze_source,
            custody_receipt_path=custody_source,
            acquisition_receipt_path=acquisition_source,
            development_root=development,
            worker_plan_path=worker_plan_path,
            worker=worker,
            formal=formal,
        )
        official_binding_receipt = _verify_official_binding_receipt(worker)
        plus = load_provider_credential(plus_env_file, channel_id=plus_channel_id)
        pro = load_provider_credential(pro_env_file, channel_id=pro_channel_id)
        if formal:
            assert formal_runtime is not None
            from replication_runtime.musique_official_hipporag_v1.binding import (
                verify_live_binding,
            )

            verify_live_binding(
                binding_receipt_path=official_binding_receipt,
                runtime_python=formal_runtime.runtime_python,
                local_llm_model=formal_runtime.local_llm_model,
                local_embedding_model=formal_runtime.local_embedding_model,
            )
            transport: ModelTransport = UrllibOpenAICompatibleTransport()
            official_retrieve = _formal_adapter(
                runtime_python=formal_runtime.runtime_python,
                local_llm_model=formal_runtime.local_llm_model,
                local_embedding_model=formal_runtime.local_embedding_model,
            )
        else:
            assert synthetic_transport is not None
            assert synthetic_official_retrieve is not None
            transport = synthetic_transport
            official_retrieve = synthetic_official_retrieve

        _consume_global_authorization(
            development_root=development,
            worker=worker,
            public_freeze=public_freeze,
        )
        authorization_consumed = True
        destination.mkdir(mode=0o700)
        os.chmod(destination, 0o700)
        precommit = _write_provider_precommit(
            output_root=destination, plus=plus, pro=pro, formal=formal
        )
        external_ruoli_calls_started = True
        selected_label, selected, selection, canary_calls = _select_provider(
            output_root=destination,
            plus=plus,
            pro=pro,
            transport=transport,
            precommit=precommit,
        )
        typed_binding = worker["typed_binding"]
        typed_program_path = _safe_relative_file(
            development,
            typed_binding["frozen_program_relative_path"],
            "typed program",
        )
        typed_receipt_path = _safe_relative_file(
            development,
            typed_binding["formation_receipt_relative_path"],
            "typed receipt",
        )
        official_retrieval_root = destination / "official_retrieval"
        official_retrieval_root.mkdir(mode=0o700)

        def prepare_one(item: Mapping[str, Any]) -> tuple[PreparedWork, ...]:
            return _prepare_item(
                development_root=development,
                item=item,
                typed_binding=typed_binding,
                typed_program_path=typed_program_path,
                typed_receipt_path=typed_receipt_path,
                official_retrieve=official_retrieve,
                official_work_root=official_retrieval_root
                / stable_hash({"anonymous_item_id": item["anonymous_item_id"]}),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=ITEM_COUNT) as executor:
            item_futures = tuple(
                executor.submit(prepare_one, item) for item in worker["items"]
            )
            prepared = tuple(
                sorted(
                    (work for future in item_futures for work in future.result()),
                    key=lambda work: work.work_unit_id,
                )
            )
        if (
            len(prepared) != WORK_UNIT_COUNT
            or {work.arm for work in prepared} != set(ARM_IDS)
            or len({work.work_unit_id for work in prepared}) != WORK_UNIT_COUNT
        ):
            raise MuSiQueFormalRunnerError("prepared generation grid drifted")
        request_hashes = {
            work.work_unit_id: work.request.request_hash for work in prepared
        }
        launch = atomic_write_hashed_json_v2(
            destination / LAUNCH_FILENAME,
            {
                "all_request_hashes_persisted_before_generator_calls": True,
                "attempts_per_work_unit": 1,
                "authorization_id": worker["authorization_id"],
                "launch_version": LAUNCH_VERSION,
                "maximum_model_concurrency": MAXIMUM_MODEL_CONCURRENCY,
                "model": MODEL_ID,
                "model_request_body_byte_budget": MODEL_REQUEST_BODY_BYTE_BUDGET,
                "request_hash_count": len(request_hashes),
                "request_hashes": request_hashes,
                "request_hash_set_sha256": stable_hash(request_hashes),
                "replay_authorized": False,
                "resampling_authorized": False,
                "retry_authorized": False,
                "selection_hash": selection["selection_hash"],
                "worker_plan_hash": worker["worker_plan_hash"],
            },
            hash_field="launch_hash",
        )
        claims = _preclaim(
            output_root=destination,
            prepared=prepared,
            selection_hash=str(selection["selection_hash"]),
        )
        barrier = threading.Barrier(WORK_UNIT_COUNT)
        concurrency = _ConcurrencyCounter()

        def run_one(work: PreparedWork) -> dict[str, Any]:
            return _run_generator_once(
                output_root=destination,
                work=work,
                claim=claims[work.work_unit_id],
                selected_label=selected_label,
                credential=selected,
                transport=transport,
                start_barrier=barrier,
                concurrency=concurrency,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=WORK_UNIT_COUNT
        ) as executor:
            generation_work_may_have_started = True
            futures = tuple(executor.submit(run_one, work) for work in prepared)
            terminals_list = tuple(future.result() for future in futures)
        terminals = {
            str(terminal["work_unit_id"]): terminal for terminal in terminals_list
        }
        if len(terminals) != WORK_UNIT_COUNT:
            raise MuSiQueNoReplayError(
                "the generation grid did not reach 18 terminal joins"
            )
        private_index = _safe_relative_file(
            development,
            worker["private_index_binding"]["private_index_relative_path"],
            "private evaluation index",
        )
        gold_open_may_have_started = True
        private_evaluation_may_have_started = True
        offline, private_evaluation = _offline_evaluate(
            output_root=destination,
            private_index_path=private_index,
            private_index_binding=worker["private_index_binding"],
            prepared=prepared,
            terminals=terminals,
        )
        task_model_calls = sum(
            int(terminal.get("model_call_count") == 1)
            for terminal in terminals.values()
        )
        execution_integrity_valid = (
            task_model_calls == WORK_UNIT_COUNT
            and len(terminals) == WORK_UNIT_COUNT
            and all(
                terminal.get("execution_terminal") is True
                and terminal.get("attempt_count") == 1
                for terminal in terminals.values()
            )
        )
        paired_evidence_complete = (
            offline["oracle_disagreement_count"] == 0
            and all(
                metric_counts["paired_count"] == ITEM_COUNT
                and metric_counts["gain_count"]
                + metric_counts["harm_count"]
                + metric_counts["tie_count"]
                == ITEM_COUNT
                for contrast in offline["pairwise_item_counts"].values()
                for metric_counts in contrast.values()
            )
        )
        formal_evidence_valid = (
            formal and execution_integrity_valid and paired_evidence_complete
        )
        body = {
            "authorization": {
                "authorization_consumed": True,
                "authorization_id": worker["authorization_id"],
                "execution_root_commitment": worker["execution_root_commitment"],
                "global_marker_persisted": True,
                "replay_authorized": False,
            },
            "call_ledger": {
                "offline_local_activity": {
                    "counted_as_ruoli_external_calls": False,
                    "official_hipporag_local_causal_model_activity": formal,
                    "official_hipporag_local_embedding_activity": formal,
                    "official_hipporag_retrieve_only_item_calls": ITEM_COUNT,
                    "typed_retriever_local_calls": ITEM_COUNT,
                },
                "offline_oracle_calls": {
                    "answer_oracle_calls": offline["answer_oracle_call_count"],
                    "support_oracle_calls": offline["support_oracle_call_count"],
                },
                "online_evaluator_calls": 0,
                "replays": 0,
                "resamples": 0,
                "retries": 0,
                "ruoli_external_calls": {
                    "constant_canary_calls": canary_calls,
                    "generator_calls": task_model_calls,
                    "total": canary_calls + task_model_calls,
                },
            },
            "concurrency": {
                "all_18_claims_persisted_before_work_start": True,
                "all_18_futures_submitted_before_results_read": True,
                "configured_model_concurrency": WORK_UNIT_COUNT,
                "observed_maximum_model_calls": concurrency.maximum,
            },
            "content_boundary": {
                "answer_aliases_persisted_publicly": False,
                "development_input_persisted_publicly": False,
                "generator_prediction_persisted_publicly": False,
                "private_index_bytes_hashed_before_generation": True,
                "private_index_parsed_only_after_terminal_join": True,
                "private_index_persisted_publicly": False,
                "questions_or_contexts_persisted_publicly": False,
                "support_labels_persisted_publicly": False,
                "trace_persisted_publicly": False,
            },
            "descriptive_only": True,
            "execution_completed": True,
            "execution_integrity_valid": execution_integrity_valid,
            "formal_evidence": formal,
            "formal_evidence_valid": formal_evidence_valid,
            "generator_terminal_join_count": len(terminals),
            "launch_hash": launch["launch_hash"],
            "offline_evaluation": offline,
            "offline_evaluation_only": True,
            "paired_evidence_complete": paired_evidence_complete,
            "performance_gate_applied": False,
            "private_evaluation_hash": private_evaluation["evaluation_hash"],
            "promotion_authorized": False,
            "provider_precommit_hash": precommit["precommit_hash"],
            "public_freeze_hash": public_freeze["freeze_sha256"],
            "public_report_version": PUBLIC_REPORT_VERSION,
            "runner_version": RUNNER_VERSION,
            "sealed_content_accessed": False,
            "selected_provider_fixed_for_complete_batch": True,
            "selected_provider_label": selected_label,
            "selection_hash": selection["selection_hash"],
            "study_id": worker["study_id"],
            "worker_plan_hash": worker["worker_plan_hash"],
        }
        _assert_public_aggregate_only(body, (plus, pro))
        return atomic_write_hashed_json_v2(
            destination / PUBLIC_REPORT_FILENAME, body, hash_field="report_hash"
        )
    except BaseException as exc:
        if worker is not None:
            _write_global_failure_receipt(
                development_root=development,
                worker=worker,
                exc=exc,
                authorization_consumed=authorization_consumed,
                external_ruoli_calls_started=external_ruoli_calls_started,
                generation_work_may_have_started=generation_work_may_have_started,
                gold_open_may_have_started=gold_open_may_have_started,
                private_evaluation_may_have_started=(
                    private_evaluation_may_have_started
                ),
            )
        raise


def run_formal_musique_three_arm(
    *,
    development_root: str | Path,
    public_freeze_path: str | Path,
    custody_receipt_path: str | Path,
    acquisition_receipt_path: str | Path,
    output_root: str | Path,
    plus_env_file: str | Path,
    pro_env_file: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
) -> dict[str, Any]:
    """Run formal evidence with fixed transport and fixed official adapter."""

    runtime = Path(runtime_python).resolve(strict=True)
    llm = Path(local_llm_model).resolve(strict=True)
    embedding = Path(local_embedding_model).resolve(strict=True)
    return _run_core(
        development_root=development_root,
        public_freeze_path=public_freeze_path,
        custody_receipt_path=custody_receipt_path,
        acquisition_receipt_path=acquisition_receipt_path,
        output_root=output_root,
        plus_env_file=plus_env_file,
        pro_env_file=pro_env_file,
        plus_channel_id=plus_channel_id,
        pro_channel_id=pro_channel_id,
        formal_runtime=_FormalRuntimePaths(
            runtime_python=runtime,
            local_llm_model=llm,
            local_embedding_model=embedding,
        ),
    )


def run_synthetic_musique_three_arm_for_tests(
    *,
    development_root: str | Path,
    public_freeze_path: str | Path,
    custody_receipt_path: str | Path,
    acquisition_receipt_path: str | Path,
    output_root: str | Path,
    plus_env_file: str | Path,
    pro_env_file: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
    transport: ModelTransport,
    official_retrieve: OfficialRetrieveOnly,
) -> dict[str, Any]:
    """Run protocol tests with injected dependencies; never formal evidence."""

    return _run_core(
        development_root=development_root,
        public_freeze_path=public_freeze_path,
        custody_receipt_path=custody_receipt_path,
        acquisition_receipt_path=acquisition_receipt_path,
        output_root=output_root,
        plus_env_file=plus_env_file,
        pro_env_file=pro_env_file,
        plus_channel_id=plus_channel_id,
        pro_channel_id=pro_channel_id,
        synthetic_transport=transport,
        synthetic_official_retrieve=official_retrieve,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--public-freeze", required=True)
    parser.add_argument("--custody-receipt", required=True)
    parser.add_argument("--acquisition-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plus-env", required=True)
    parser.add_argument("--pro-env", required=True)
    parser.add_argument("--plus-channel-id", required=True)
    parser.add_argument("--pro-channel-id", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--local-llm-model", required=True)
    parser.add_argument("--local-embedding-model", required=True)
    arguments = parser.parse_args(argv)
    report = run_formal_musique_three_arm(
        development_root=arguments.development_root,
        public_freeze_path=arguments.public_freeze,
        custody_receipt_path=arguments.custody_receipt,
        acquisition_receipt_path=arguments.acquisition_receipt,
        output_root=arguments.output_root,
        plus_env_file=arguments.plus_env,
        pro_env_file=arguments.pro_env,
        plus_channel_id=arguments.plus_channel_id,
        pro_channel_id=arguments.pro_channel_id,
        runtime_python=arguments.runtime_python,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    print(
        json.dumps(
            {
                "execution_completed": report["execution_completed"],
                "formal_evidence_valid": report["formal_evidence_valid"],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_IDS",
    "CONTEXT_SERIALIZATION_VERSION",
    "ITEM_COUNT",
    "ITEM_INPUT_VERSION",
    "MODEL_ID",
    "MODEL_OUTPUT_TOKEN_BUDGET",
    "MODEL_REQUEST_BODY_BYTE_BUDGET",
    "MuSiQueFormalRunnerError",
    "MuSiQueNoReplayError",
    "OFFICIAL_ADAPTER_ID",
    "OFFICIAL_BINDING_RECEIPT_RELATIVE",
    "PRIVATE_INDEX_VERSION",
    "PROMPT_VERSION",
    "RUNNER_VERSION",
    "TOP_K",
    "WORKER_PLAN_VERSION",
    "WORK_UNIT_COUNT",
    "current_runner_implementation_binding",
    "main",
    "provider_identity_binding",
    "run_formal_musique_three_arm",
    "run_synthetic_musique_three_arm_for_tests",
]
