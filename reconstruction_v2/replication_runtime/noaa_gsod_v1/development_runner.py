from __future__ import annotations

"""Execute the frozen NOAA development grid exactly once.

The runner deliberately keeps provider selection separate from task content:
both provider identities are committed first, a constant non-task canary picks
one complete route, and only then are the private worker/controller plans read.
All task outputs and item-level evaluation receipts stay below the ignored
development root.  The aggregate report is safe to publish.
"""

import argparse
import concurrent.futures
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
    read_hashed_json_v2,
)

from . import oracle_sqlite, oracle_stdlib
from .contract import (
    ORACLE_IDS,
    STUDY_ID,
    TASK_CONTRACT,
    NoaaGsodError,
    canonical_json_bytes,
    payload_hash,
    sha256_file,
)
from .development_freeze import (
    ARM_IDS,
    DEVELOPMENT_ITEM_COUNT,
    MODEL_ID,
    MODEL_OUTPUT_TOKEN_BUDGET,
    MODEL_REQUEST_BODY_BYTE_BUDGET,
    MODEL_WORK_UNIT_COUNT,
    ProviderIdentity,
    WORK_UNIT_COUNT,
    endpoint_identity_hash as frozen_endpoint_identity_hash,
    verify_controller_plan,
    verify_public_pre_run_freeze,
    verify_worker_plan,
)
from .development_implementation import build_development_implementation_set
from .development_schemas import DEVELOPMENT_SCHEMA_SET_HASH
from .pack import read_json
from .typed_relational import (
    OPERATOR_VERSION,
    TypedRelationalProgram,
    execute_frozen_operator,
    load_frozen_program,
)


RUNNER_VERSION = "noaa_gsod_formal_development_runner_v1"
PROVIDER_PRECOMMIT_VERSION = "noaa_gsod_provider_identity_precommit_v1"
PROVIDER_SELECTION_VERSION = "noaa_gsod_provider_selection_v1"
CANARY_VERSION = "noaa_gsod_constant_transport_canary_v1"
LAUNCH_TRANSITION_VERSION = "noaa_gsod_development_launch_transition_v1"
WORK_CLAIM_VERSION = "noaa_gsod_development_work_claim_v1"
WORK_CLAIM_SET_VERSION = "noaa_gsod_development_work_claim_set_v1"
WORK_TERMINAL_VERSION = "noaa_gsod_development_work_terminal_v1"
PRIVATE_EVALUATION_VERSION = "noaa_gsod_private_offline_evaluation_v1"
API_KEY_COMMITMENT_VERSION = "noaa_gsod_api_key_hmac_sha256_v1"
API_KEY_COMMITMENT_CHALLENGE = (
    b"noaa-gsod/formal-development/provider-key/pre-batch/v1"
)
PROVIDER_API_ORIGIN = "https://ruoli.dev"
FORMAL_MODEL_TRANSPORT_ID = "urllib_openai_compatible_chat_completions_v1"
SYNTHETIC_INJECTED_DEPENDENCY_ID = "synthetic_injected_unattested_v1"

WORKER_PLAN_FILENAME = "worker_plan.json"
CONTROLLER_PLAN_FILENAME = "controller_plan.private.json"
PROVIDER_PRECOMMIT_FILENAME = "provider.identity.precommit.json"
PROVIDER_SELECTION_FILENAME = "provider.selection.json"
LAUNCH_FILENAME = "batch.launch.json"
WORK_CLAIM_SET_FILENAME = "work.claim-set.json"
PRIVATE_EVALUATION_FILENAME = "evaluation.private.json"
REPORT_FILENAME = "development.report.json"

CANARY_SYSTEM_PROMPT = (
    "This is a fixed transport-only availability canary. Do not use tools, "
    "files, network search, or external context. Return any non-empty response."
)
CANARY_PAYLOAD: dict[str, Any] = {
    "canary_version": CANARY_VERSION,
    "contains_task_or_development_content": False,
    "fixed_probe": "transport_complete_response_only",
}
RAW_SYSTEM_PROMPT = (
    "Solve the supplied frozen NOAA CSV task using only the supplied contract "
    "and CSV. Return exactly the canonical three-field JSON object and nothing "
    "else. Do not use tools, files, or network access."
)
AGENT_SYSTEM_PROMPT = (
    "Select the applicable frozen typed operator. Return exactly one canonical "
    "JSON object whose only field is program_id and whose value is one of the "
    "opaque IDs supplied. Do not return an answer, explanation, markdown, or "
    "extra field. Do not use tools, files, or network access."
)

_ENV_ALLOWLIST = frozenset(
    {
        "ASSUMPTION_V2_API_BASE",
        "ASSUMPTION_V2_API_KEY",
        "ASSUMPTION_V2_MODEL",
        "ASSUMPTION_V2_PROVIDER_CHAIN",
    }
)
_OUTPUT_FIELDS = frozenset(
    {"mean_daily_precip_mm", "month", "valid_day_count"}
)
_MONTH = re.compile(r"^(?:0[1-9]|1[0-2])$")
_FIXED_TWO_DECIMALS = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{2}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ENV_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 1024 * 1024
_MODEL_TIMEOUT_SECONDS = 300.0


class DevelopmentRunnerError(RuntimeError):
    """The formal execution boundary failed closed."""


class ProviderTransportUnavailable(DevelopmentRunnerError):
    """No complete model response was available for transport reasons."""


class ProviderProtocolError(DevelopmentRunnerError):
    """The provider responded, but its wire response was malformed."""


class OutputContractError(DevelopmentRunnerError):
    """A model response violated its exact frozen output contract."""


class NoReplayError(DevelopmentRunnerError):
    """A consumed authorization lacks a terminal receipt and cannot be replayed."""


def endpoint_identity_hash(api_origin: str) -> str:
    return frozen_endpoint_identity_hash(api_origin)


def api_key_hmac_commitment(api_key: str) -> str:
    normalized = str(api_key).strip()
    if not normalized:
        raise DevelopmentRunnerError("provider API key is absent")
    return hmac.new(
        normalized.encode("utf-8"),
        API_KEY_COMMITMENT_CHALLENGE,
        hashlib.sha256,
    ).hexdigest()


def _canonical_provider_base(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(value.strip())
        port_value = parsed.port
    except ValueError as exc:
        raise DevelopmentRunnerError("provider API base is malformed") from exc
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or path not in {"", "/v1"}
    ):
        raise DevelopmentRunnerError("provider API base is not an allowed origin")
    port = f":{port_value}" if port_value is not None else ""
    origin = f"{parsed.scheme}://{parsed.hostname}{port}"
    return origin + path, origin


def _repository_root() -> Path | None:
    project = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return Path(value).resolve(strict=True) if value else None


def _require_ignored_or_external(path: Path) -> None:
    repository = _repository_root()
    if repository is None:
        return
    try:
        path.relative_to(repository)
    except ValueError:
        return
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(path)],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise DevelopmentRunnerError(
            "provider env inside the repository is not git-ignored"
        )


@dataclass(frozen=True)
class ProviderCredential:
    channel_id: str
    api_base: str
    api_origin: str
    api_key: str = field(repr=False, compare=False)
    model: str = MODEL_ID

    @property
    def endpoint_identity_hash(self) -> str:
        return endpoint_identity_hash(self.api_origin)

    @property
    def api_key_hmac_sha256(self) -> str:
        return api_key_hmac_commitment(self.api_key)

    def safe_identity(self, *, provider_label: str) -> dict[str, Any]:
        if provider_label not in {"plus", "pro"}:
            raise DevelopmentRunnerError("provider label is not preregistered")
        return {
            "api_key_commitment_version": API_KEY_COMMITMENT_VERSION,
            "api_key_hmac_sha256": self.api_key_hmac_sha256,
            "api_origin": self.api_origin,
            "channel_id": self.channel_id,
            "endpoint_identity_hash": self.endpoint_identity_hash,
            "model": self.model,
            "provider_label": provider_label,
            "provider_chain": "openai_compatible",
            "secret_value_persisted": False,
        }


def load_provider_credential(
    env_file: str | Path, *, channel_id: str
) -> ProviderCredential:
    unresolved = Path(env_file).expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise DevelopmentRunnerError("provider env is not a regular file")
    source = unresolved.resolve(strict=True)
    mode = stat.S_IMODE(source.stat().st_mode)
    if mode != 0o600:
        raise DevelopmentRunnerError("provider env must have mode 0600")
    if source.stat().st_size > _MAX_ENV_BYTES:
        raise DevelopmentRunnerError("provider env exceeds its byte bound")
    _require_ignored_or_external(source)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DevelopmentRunnerError("provider env is unreadable") from exc
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise DevelopmentRunnerError("provider env contains a malformed row")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in _ENV_ALLOWLIST or key in values:
            raise DevelopmentRunnerError(
                "provider env contains an unapproved or duplicate key"
            )
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise DevelopmentRunnerError("provider env contains an empty value")
        values[key] = value
    if set(values) != _ENV_ALLOWLIST:
        raise DevelopmentRunnerError("provider env does not contain the exact allowlist")
    if values["ASSUMPTION_V2_MODEL"] != MODEL_ID:
        raise DevelopmentRunnerError("provider model differs from the frozen model")
    if values["ASSUMPTION_V2_PROVIDER_CHAIN"] != "openai_compatible":
        raise DevelopmentRunnerError("provider chain differs from the frozen route")
    api_base, api_origin = _canonical_provider_base(
        values["ASSUMPTION_V2_API_BASE"]
    )
    return ProviderCredential(
        channel_id=channel_id,
        api_base=api_base,
        api_origin=api_origin,
        api_key=values["ASSUMPTION_V2_API_KEY"],
    )


@dataclass(frozen=True)
class ModelRequest:
    purpose: str
    system_prompt: str = field(repr=False)
    user_payload: Mapping[str, Any] = field(repr=False)
    max_output_tokens: int = MODEL_OUTPUT_TOKEN_BUDGET
    json_object: bool = True

    def body(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"content": self.system_prompt, "role": "system"},
                {
                    "content": json.dumps(
                        self.user_payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "role": "user",
                },
            ],
            "model": MODEL_ID,
            "temperature": 0,
        }
        if self.json_object:
            result["response_format"] = {"type": "json_object"}
        return result

    @property
    def request_hash(self) -> str:
        return payload_hash({"purpose": self.purpose, "body": self.body()})

    @property
    def canonical_body_byte_count(self) -> int:
        return len(canonical_json_bytes(self.body()))


class ModelTransport(Protocol):
    def complete(
        self, *, credential: ProviderCredential, request: ModelRequest
    ) -> str: ...


class UrllibOpenAICompatibleTransport:
    """One-attempt OpenAI-compatible chat-completions transport."""

    def complete(
        self, *, credential: ProviderCredential, request: ModelRequest
    ) -> str:
        base = credential.api_base.rstrip("/")
        endpoint = (
            f"{base}/chat/completions"
            if base.endswith("/v1")
            else f"{base}/v1/chat/completions"
        )
        wire_request = urllib.request.Request(
            endpoint,
            data=json.dumps(request.body(), ensure_ascii=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {credential.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                wire_request, timeout=_MODEL_TIMEOUT_SECONDS
            ) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            canary_auth_without_response = (
                request.purpose == "provider_transport_canary"
                and exc.code in {401, 403}
            )
            if (
                canary_auth_without_response
                or exc.code in {408, 429}
                or 500 <= exc.code < 600
            ):
                raise ProviderTransportUnavailable(
                    "provider transport returned an unavailable status"
                ) from exc
            raise ProviderProtocolError(
                "provider transport returned a non-fallback status"
            ) from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ProviderTransportUnavailable(
                "provider transport did not complete"
            ) from exc
        if not raw or len(raw) > _MAX_RESPONSE_BYTES:
            raise ProviderProtocolError("provider response is empty or oversized")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError("provider wire JSON is malformed") from exc
        content = _message_content(payload)
        if not content.strip():
            raise ProviderProtocolError("provider response has no non-empty content")
        return content


def _message_content(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise ProviderProtocolError("provider response root is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderProtocolError("provider response has no choice")
    first = choices[0]
    if not isinstance(first, Mapping) or not isinstance(
        first.get("message"), Mapping
    ):
        raise ProviderProtocolError("provider response message is malformed")
    content = first["message"].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, Mapping)
        )
    raise ProviderProtocolError("provider response content is malformed")


def _write_or_verify_receipt(
    path: Path, body: Mapping[str, Any], *, hash_field: str
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        observed = read_hashed_json_v2(path, hash_field=hash_field)
        observed_body = dict(observed)
        observed_body.pop(hash_field, None)
        if observed_body != dict(body):
            raise DevelopmentRunnerError("durable receipt differs from frozen state")
        return observed
    return atomic_write_hashed_json_v2(path, body, hash_field=hash_field)


def _execution_dependency_ids(*, dependency_injection_used: bool) -> dict[str, Any]:
    if dependency_injection_used:
        return {
            "local_operator": OPERATOR_VERSION,
            "model_transport": SYNTHETIC_INJECTED_DEPENDENCY_ID,
            "offline_oracles": [SYNTHETIC_INJECTED_DEPENDENCY_ID],
        }
    return {
        "local_operator": OPERATOR_VERSION,
        "model_transport": FORMAL_MODEL_TRANSPORT_ID,
        "offline_oracles": list(ORACLE_IDS),
    }


def _provider_precommit(
    *,
    output_root: Path,
    public_freeze: Mapping[str, Any],
    plus: ProviderCredential,
    pro: ProviderCredential,
    dependency_injection_used: bool,
    execution_dependency_ids: Mapping[str, Any],
    formal_evidence: bool,
) -> dict[str, Any]:
    if (
        plus.api_origin != PROVIDER_API_ORIGIN
        or pro.api_origin != PROVIDER_API_ORIGIN
        or plus.model != MODEL_ID
        or pro.model != MODEL_ID
    ):
        raise DevelopmentRunnerError(
            "provider registration differs from the exact origin/model contract"
        )
    if plus.api_key_hmac_sha256 == pro.api_key_hmac_sha256:
        raise DevelopmentRunnerError(
            "Plus and Pro must use distinct API-key HMAC commitments"
        )
    provider_identity = ProviderIdentity(
        plus_channel_id=plus.channel_id,
        plus_endpoint_origin=plus.api_origin,
        pro_channel_id=pro.channel_id,
        pro_endpoint_origin=pro.api_origin,
    )
    provider_identity.validate()
    expected_identity_hash = public_freeze["binding_hashes"].get(
        "provider_identity_hash"
    )
    if provider_identity.identity_hash != expected_identity_hash:
        raise DevelopmentRunnerError(
            "provider endpoint/channel identity differs from the pre-run freeze"
        )
    body = {
        "api_key_persisted": False,
        "canary_request_hash": ModelRequest(
            purpose="provider_transport_canary",
            system_prompt=CANARY_SYSTEM_PROMPT,
            user_payload=CANARY_PAYLOAD,
            max_output_tokens=32,
            json_object=False,
        ).request_hash,
        "canary_uses_development_input": False,
        "dependency_injection_used": dependency_injection_used,
        "execution_dependency_ids": dict(execution_dependency_ids),
        "formal_evidence": formal_evidence,
        "precommit_version": PROVIDER_PRECOMMIT_VERSION,
        "provider_identity_hash": provider_identity.identity_hash,
        "provider_registration_contract": {
            "api_origin": PROVIDER_API_ORIGIN,
            "model": MODEL_ID,
            "provider_labels": ["plus", "pro"],
        },
        "providers": {
            "plus": plus.safe_identity(provider_label="plus"),
            "pro": pro.safe_identity(provider_label="pro"),
        },
        "secret_value_persisted": False,
        "task_calls_before_precommit": 0,
    }
    return _write_or_verify_receipt(
        output_root / PROVIDER_PRECOMMIT_FILENAME,
        body,
        hash_field="precommit_hash",
    )


def _canary_paths(output_root: Path, label: str) -> tuple[Path, Path, Path]:
    root = output_root / "provider_canary"
    return (
        root / f"{label}.claim.json",
        root / f"{label}.success.json",
        root / f"{label}.failure.json",
    )


def _read_canary_outcome(
    *,
    claim_path: Path,
    success_path: Path,
    failure_path: Path,
) -> dict[str, Any] | None:
    claim_exists = claim_path.exists() or claim_path.is_symlink()
    success_exists = success_path.exists() or success_path.is_symlink()
    failure_exists = failure_path.exists() or failure_path.is_symlink()
    if success_exists and failure_exists:
        raise DevelopmentRunnerError("provider canary has two terminal outcomes")
    if success_exists or failure_exists:
        if not claim_exists:
            raise DevelopmentRunnerError("provider canary terminal lacks its claim")
        claim = read_hashed_json_v2(claim_path, hash_field="claim_hash")
        terminal = read_hashed_json_v2(
            success_path if success_exists else failure_path,
            hash_field="receipt_hash",
        )
        if terminal.get("claim_hash") != claim.get("claim_hash"):
            raise DevelopmentRunnerError("provider canary claim binding drifted")
        return terminal
    if claim_exists:
        raise NoReplayError("provider canary claim is incomplete; replay forbidden")
    return None


def _run_canary_once(
    *,
    output_root: Path,
    label: str,
    credential: ProviderCredential,
    precommit_hash: str,
    transport: ModelTransport,
) -> dict[str, Any]:
    claim_path, success_path, failure_path = _canary_paths(output_root, label)
    existing = _read_canary_outcome(
        claim_path=claim_path,
        success_path=success_path,
        failure_path=failure_path,
    )
    if existing is not None:
        return existing
    request = ModelRequest(
        purpose="provider_transport_canary",
        system_prompt=CANARY_SYSTEM_PROMPT,
        user_payload=CANARY_PAYLOAD,
        max_output_tokens=32,
        json_object=False,
    )
    claim = atomic_write_hashed_json_v2(
        claim_path,
        {
            "attempts": 1,
            "canary_version": CANARY_VERSION,
            "model": MODEL_ID,
            "model_replay_authorized": False,
            "precommit_hash": precommit_hash,
            "provider_identity_hash": payload_hash(
                credential.safe_identity(provider_label=label)
            ),
            "provider_label": label,
            "request_hash": request.request_hash,
            "task_or_development_content_accessed": False,
        },
        hash_field="claim_hash",
    )
    try:
        content = transport.complete(credential=credential, request=request)
        if not isinstance(content, str) or not content.strip():
            raise ProviderProtocolError("provider canary content is incomplete")
    except ProviderTransportUnavailable:
        return atomic_write_hashed_json_v2(
            failure_path,
            {
                "attempts": 1,
                "claim_hash": claim["claim_hash"],
                "failure_kind": "transport_unavailable",
                "model_call_count": 1,
                "model_response_received": False,
                "raw_failure_persisted": False,
                "semantic_acceptance_used_for_selection": False,
                "secret_value_persisted": False,
            },
            hash_field="receipt_hash",
        )
    except Exception as exc:
        terminal = atomic_write_hashed_json_v2(
            failure_path,
            {
                "attempts": 1,
                "claim_hash": claim["claim_hash"],
                "failure_kind": "provider_protocol_failure",
                "model_call_count": 1,
                "model_response_received": False,
                "raw_failure_persisted": False,
                "semantic_acceptance_used_for_selection": False,
                "secret_value_persisted": False,
            },
            hash_field="receipt_hash",
        )
        raise ProviderProtocolError(
            "provider canary failed for a non-transport reason"
        ) from exc
    return atomic_write_hashed_json_v2(
        success_path,
        {
            "attempts": 1,
            "claim_hash": claim["claim_hash"],
            "failure_kind": None,
            "model_call_count": 1,
            "model_response_received": True,
            "raw_response_persisted": False,
            "response_hash": payload_hash({"content": content}),
            "semantic_acceptance_used_for_selection": False,
            "secret_value_persisted": False,
        },
        hash_field="receipt_hash",
    )


def _validate_existing_selection(
    selection: Mapping[str, Any], *, precommit_hash: str, output_root: Path
) -> str:
    label = selection.get("selected_provider_label")
    if label not in {"plus", "pro"}:
        raise DevelopmentRunnerError("selected provider label is invalid")
    body = dict(selection)
    declared = body.pop("selection_hash", None)
    if (
        declared != payload_hash(body)
        or selection.get("selection_version") != PROVIDER_SELECTION_VERSION
        or selection.get("precommit_hash") != precommit_hash
        or selection.get("selection_completed_before_task_calls") is not True
        or selection.get("task_calls_before_selection") != 0
        or selection.get("selected_provider_fixed_for_complete_batch") is not True
        or selection.get("mid_batch_provider_switch_authorized") is not False
        or selection.get("retry_authorized") is not False
        or selection.get("resampling_authorized") is not False
        or selection.get("semantic_acceptance_used_for_selection") is not False
        or selection.get("raw_canary_content_persisted") is not False
        or selection.get("secret_value_persisted") is not False
    ):
        raise DevelopmentRunnerError("provider selection receipt drifted")
    plus = read_hashed_json_v2(
        _canary_paths(output_root, "plus")[
            1 if label == "plus" else 2
        ],
        hash_field="receipt_hash",
    )
    if label == "plus":
        if (
            plus.get("model_response_received") is not True
            or selection.get("probe_order") != ["plus_complete_model_response"]
            or selection.get("plus_canary_receipt_hash")
            != plus.get("receipt_hash")
            or selection.get("pro_canary_receipt_hash") is not None
        ):
            raise DevelopmentRunnerError("Plus provider selection drifted")
    else:
        pro = read_hashed_json_v2(
            _canary_paths(output_root, "pro")[1], hash_field="receipt_hash"
        )
        if (
            plus.get("failure_kind") != "transport_unavailable"
            or pro.get("model_response_received") is not True
            or selection.get("probe_order")
            != [
                "plus_transport_unavailable",
                "pro_complete_model_response",
            ]
            or selection.get("plus_canary_receipt_hash")
            != plus.get("receipt_hash")
            or selection.get("pro_canary_receipt_hash")
            != pro.get("receipt_hash")
        ):
            raise DevelopmentRunnerError("Pro fallback selection drifted")
    return str(label)


def select_provider(
    *,
    output_root: Path,
    precommit: Mapping[str, Any],
    plus: ProviderCredential,
    pro: ProviderCredential,
    transport: ModelTransport,
) -> tuple[str, ProviderCredential, dict[str, Any]]:
    selection_path = output_root / PROVIDER_SELECTION_FILENAME
    if selection_path.exists() or selection_path.is_symlink():
        selection = read_hashed_json_v2(
            selection_path, hash_field="selection_hash"
        )
        label = _validate_existing_selection(
            selection,
            precommit_hash=str(precommit["precommit_hash"]),
            output_root=output_root,
        )
        return label, plus if label == "plus" else pro, selection

    plus_outcome = _run_canary_once(
        output_root=output_root,
        label="plus",
        credential=plus,
        precommit_hash=str(precommit["precommit_hash"]),
        transport=transport,
    )
    if plus_outcome.get("model_response_received") is True:
        label = "plus"
        selected = plus
        probe_order = ["plus_complete_model_response"]
        pro_receipt_hash = None
    elif plus_outcome.get("failure_kind") == "transport_unavailable":
        pro_outcome = _run_canary_once(
            output_root=output_root,
            label="pro",
            credential=pro,
            precommit_hash=str(precommit["precommit_hash"]),
            transport=transport,
        )
        if pro_outcome.get("model_response_received") is not True:
            raise ProviderTransportUnavailable(
                "Pro canary did not return a complete response"
            )
        label = "pro"
        selected = pro
        probe_order = [
            "plus_transport_unavailable",
            "pro_complete_model_response",
        ]
        pro_receipt_hash = pro_outcome["receipt_hash"]
    else:
        raise ProviderProtocolError(
            "Plus canary failed without an authorized transport fallback"
        )
    body = {
        "mid_batch_provider_switch_authorized": False,
        "precommit_hash": precommit["precommit_hash"],
        "probe_order": probe_order,
        "pro_canary_receipt_hash": pro_receipt_hash,
        "plus_canary_receipt_hash": plus_outcome["receipt_hash"],
        "raw_canary_content_persisted": False,
        "resampling_authorized": False,
        "retry_authorized": False,
        "secret_value_persisted": False,
        "selected_provider_fixed_for_complete_batch": True,
        "selected_provider_label": label,
        "selection_completed_before_task_calls": True,
        "selection_version": PROVIDER_SELECTION_VERSION,
        "semantic_acceptance_used_for_selection": False,
        "task_calls_before_selection": 0,
    }
    selection = atomic_write_hashed_json_v2(
        selection_path, body, hash_field="selection_hash"
    )
    return label, selected, selection


def _strict_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    try:
        value = json.loads(
            stripped,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid constant: {token}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OutputContractError("model response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise OutputContractError("model response root is not an object")
    if canonical_json_bytes(value).decode("utf-8") != stripped:
        raise OutputContractError("model response is not canonical JSON")
    return value


def _validate_task_output(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    mean = result.get("mean_daily_precip_mm")
    month = result.get("month")
    count = result.get("valid_day_count")
    if (
        set(result) != _OUTPUT_FIELDS
        or not isinstance(mean, str)
        or _FIXED_TWO_DECIMALS.fullmatch(mean) is None
        or not isinstance(month, str)
        or _MONTH.fullmatch(month) is None
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise OutputContractError("task output schema is invalid")
    try:
        parsed_mean = Decimal(mean)
    except InvalidOperation as exc:
        raise OutputContractError("task output mean is invalid") from exc
    if not parsed_mean.is_finite() or parsed_mean < 0:
        raise OutputContractError("task output mean is invalid")
    return result


def _parse_raw_response(content: str) -> dict[str, Any]:
    return _validate_task_output(_strict_json_object(content))


def _parse_agent_response(content: str, *, program_id: str) -> str:
    value = _strict_json_object(content)
    if set(value) != {"program_id"} or value.get("program_id") != program_id:
        raise OutputContractError("typed agent response is not the frozen program ID")
    return program_id


def _parse_operator_output(content: bytes) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeError as exc:
        raise OutputContractError("local operator output is not UTF-8") from exc
    return _validate_task_output(_strict_json_object(text))


@dataclass(frozen=True)
class PreparedWork:
    unit: Mapping[str, Any]
    unit_hash: str
    input_path: Path
    model_request: ModelRequest | None

    @property
    def work_unit_id(self) -> str:
        return str(self.unit["work_unit_id"])

    @property
    def arm(self) -> str:
        return str(self.unit["arm"])

    @property
    def anonymous_item_id(self) -> str:
        return str(self.unit["anonymous_item_id"])


@dataclass(frozen=True)
class UnitOutcome:
    prepared: PreparedWork
    terminal: Mapping[str, Any] | None
    recovered: bool
    blocked_incomplete: bool = False


class ModelConcurrency:
    def __init__(self, *, slots: int, start_participants: int) -> None:
        self.slots = slots
        self._semaphore = threading.BoundedSemaphore(slots)
        self._barrier = (
            threading.Barrier(start_participants)
            if start_participants > 1
            else None
        )
        self._lock = threading.Lock()
        self._active = 0
        self._maximum_active = 0

    @property
    def maximum_active(self) -> int:
        with self._lock:
            return self._maximum_active

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self._semaphore.acquire()
        try:
            if self._barrier is not None:
                self._barrier.wait(timeout=60)
            with self._lock:
                self._active += 1
                self._maximum_active = max(self._maximum_active, self._active)
            try:
                yield
            finally:
                with self._lock:
                    self._active -= 1
        finally:
            self._semaphore.release()


def _state_root(output_root: Path, work: PreparedWork) -> Path:
    return output_root / "worker_state" / work.unit_hash


def _work_paths(output_root: Path, work: PreparedWork) -> tuple[Path, Path]:
    state = _state_root(output_root, work)
    return state / "claim.json", state / "terminal.private.json"


def _work_request_hash(work: PreparedWork) -> str:
    if work.model_request is not None:
        return work.model_request.request_hash
    return payload_hash(
        {
            "input_sha256": work.unit["input_sha256"],
            "program_id": work.unit["program_id"],
            "purpose": "local_operator",
        }
    )


def _work_claim_body(
    *, work: PreparedWork, selection_hash: str
) -> dict[str, Any]:
    return {
        "attempt_count": 1,
        "claim_version": WORK_CLAIM_VERSION,
        "execution_authorization_count": 1,
        "input_sha256": work.unit["input_sha256"],
        "model_replay_authorized": False,
        "request_hash": _work_request_hash(work),
        "resampling_authorized": False,
        "retry_authorized": False,
        "selection_hash": selection_hash,
        "work_unit_hash": work.unit_hash,
        "work_unit_id": work.work_unit_id,
    }


def _has_any_work_state(output_root: Path) -> bool:
    root = output_root / "worker_state"
    if root.is_symlink():
        raise DevelopmentRunnerError("worker state root is symbolic")
    if not root.exists():
        return False
    if not root.is_dir():
        raise DevelopmentRunnerError("worker state root is not a directory")
    return next(root.iterdir(), None) is not None


def _load_terminal(
    *, output_root: Path, work: PreparedWork, selection_hash: str
) -> tuple[dict[str, Any] | None, bool]:
    claim_path, terminal_path = _work_paths(output_root, work)
    claim_exists = claim_path.exists() or claim_path.is_symlink()
    terminal_exists = terminal_path.exists() or terminal_path.is_symlink()
    if terminal_exists:
        if not claim_exists:
            raise DevelopmentRunnerError("work terminal lacks its execution claim")
        claim = read_hashed_json_v2(claim_path, hash_field="claim_hash")
        claim_body = dict(claim)
        claim_body.pop("claim_hash", None)
        if claim_body != _work_claim_body(
            work=work, selection_hash=selection_hash
        ):
            raise DevelopmentRunnerError("work execution claim drifted")
        terminal = read_hashed_json_v2(
            terminal_path, hash_field="terminal_hash"
        )
        if (
            terminal.get("terminal_version") != WORK_TERMINAL_VERSION
            or terminal.get("claim_hash") != claim.get("claim_hash")
            or terminal.get("work_unit_hash") != work.unit_hash
            or terminal.get("work_unit_id") != work.work_unit_id
            or terminal.get("arm") != work.arm
            or terminal.get("execution_terminal") is not True
            or terminal.get("attempt_count") != 1
            or terminal.get("raw_response_persisted") is not False
            or terminal.get("prompt_persisted") is not False
            or terminal.get("trace_persisted") is not False
            or terminal.get("secret_value_persisted") is not False
        ):
            raise DevelopmentRunnerError("work terminal identity drifted")
        output = terminal.get("output")
        if terminal.get("output_contract_valid") is True:
            if not isinstance(output, Mapping):
                raise DevelopmentRunnerError("valid terminal lacks output")
            validated = _validate_task_output(output)
            if terminal.get("output_hash") != payload_hash(validated):
                raise DevelopmentRunnerError("terminal output hash drifted")
        elif output is not None or terminal.get("output_hash") is not None:
            raise DevelopmentRunnerError("invalid terminal contains output")
        return terminal, False
    return None, claim_exists


def _inspect_existing_work_grid(
    *,
    output_root: Path,
    prepared: Sequence[PreparedWork],
    selection_hash: str,
) -> tuple[str, tuple[UnitOutcome, ...]]:
    state_root = output_root / "worker_state"
    expected_roots = {work.unit_hash for work in prepared}
    if state_root.is_symlink():
        raise DevelopmentRunnerError("worker state root is symbolic")
    if state_root.exists():
        if not state_root.is_dir():
            raise DevelopmentRunnerError("worker state root is not a directory")
        observed_roots = {path.name for path in state_root.iterdir()}
        if not observed_roots.issubset(expected_roots):
            raise DevelopmentRunnerError("worker state contains an unknown unit")

    claim_count = 0
    terminal_count = 0
    for work in prepared:
        claim_path, terminal_path = _work_paths(output_root, work)
        claim_exists = claim_path.exists() or claim_path.is_symlink()
        terminal_exists = terminal_path.exists() or terminal_path.is_symlink()
        if terminal_exists and not claim_exists:
            raise DevelopmentRunnerError("work terminal lacks its execution claim")
        claim_count += int(claim_exists)
        terminal_count += int(terminal_exists)

    if claim_count == 0 and terminal_count == 0:
        return "fresh", ()
    if claim_count != WORK_UNIT_COUNT or terminal_count != WORK_UNIT_COUNT:
        raise NoReplayError(
            "partial work claim/terminal grid exists; replay or fill-in forbidden"
        )

    outcomes: list[UnitOutcome] = []
    for work in prepared:
        terminal, incomplete = _load_terminal(
            output_root=output_root,
            work=work,
            selection_hash=selection_hash,
        )
        if terminal is None or incomplete:
            raise NoReplayError("complete terminal recovery grid drifted")
        outcomes.append(
            UnitOutcome(prepared=work, terminal=terminal, recovered=True)
        )
    return "complete_terminals", tuple(outcomes)


def _preclaim_complete_work_grid(
    *,
    output_root: Path,
    prepared: Sequence[PreparedWork],
    selection_hash: str,
) -> dict[str, dict[str, Any]]:
    if len(prepared) != WORK_UNIT_COUNT:
        raise DevelopmentRunnerError("preclaim grid does not contain 18 units")
    claims: dict[str, dict[str, Any]] = {}
    for work in prepared:
        claim_path, terminal_path = _work_paths(output_root, work)
        if (
            claim_path.exists()
            or claim_path.is_symlink()
            or terminal_path.exists()
            or terminal_path.is_symlink()
        ):
            raise NoReplayError("fresh preclaim encountered prior work state")
        claim = atomic_write_hashed_json_v2(
            claim_path,
            _work_claim_body(work=work, selection_hash=selection_hash),
            hash_field="claim_hash",
        )
        claims[work.work_unit_id] = claim
    if len(claims) != WORK_UNIT_COUNT:
        raise NoReplayError("complete 18-unit preclaim did not durably finish")
    claim_hashes = {
        work.work_unit_id: claims[work.work_unit_id]["claim_hash"]
        for work in prepared
    }
    _write_or_verify_receipt(
        output_root / WORK_CLAIM_SET_FILENAME,
        {
            "all_claims_persisted_before_work_start": True,
            "claim_count": WORK_UNIT_COUNT,
            "claim_hashes": claim_hashes,
            "claim_set_version": WORK_CLAIM_SET_VERSION,
            "selection_hash": selection_hash,
            "work_unit_hashes": [work.unit_hash for work in prepared],
        },
        hash_field="claim_set_hash",
    )
    return claims


def _validate_complete_work_claim_set(
    *,
    output_root: Path,
    prepared: Sequence[PreparedWork],
    selection_hash: str,
) -> dict[str, Any]:
    claim_set_path = output_root / WORK_CLAIM_SET_FILENAME
    if not claim_set_path.exists() or claim_set_path.is_symlink():
        raise NoReplayError(
            "complete claim-set receipt is absent; work replay/fill-in forbidden"
        )
    claim_hashes: dict[str, str] = {}
    for work in prepared:
        claim_path, _terminal_path = _work_paths(output_root, work)
        claim = read_hashed_json_v2(claim_path, hash_field="claim_hash")
        claim_body = dict(claim)
        claim_body.pop("claim_hash", None)
        if claim_body != _work_claim_body(
            work=work, selection_hash=selection_hash
        ):
            raise DevelopmentRunnerError("work execution claim drifted")
        claim_hashes[work.work_unit_id] = str(claim["claim_hash"])
    return _write_or_verify_receipt(
        claim_set_path,
        {
            "all_claims_persisted_before_work_start": True,
            "claim_count": WORK_UNIT_COUNT,
            "claim_hashes": claim_hashes,
            "claim_set_version": WORK_CLAIM_SET_VERSION,
            "selection_hash": selection_hash,
            "work_unit_hashes": [work.unit_hash for work in prepared],
        },
        hash_field="claim_set_hash",
    )


def _terminal_error_code(exc: Exception, *, arm: str) -> str:
    if isinstance(exc, ProviderTransportUnavailable):
        return "selected_provider_transport_unavailable"
    if isinstance(exc, ProviderProtocolError):
        return "selected_provider_protocol_failure"
    if isinstance(exc, OutputContractError):
        return (
            "raw_output_contract_invalid"
            if arm == "raw_model"
            else "agent_output_contract_invalid"
        )
    if arm in {"agent_typed_model", "operator_only_local"}:
        return "local_operator_failed"
    return "work_execution_failed"


def _run_work_once(
    *,
    output_root: Path,
    work: PreparedWork,
    claim: Mapping[str, Any],
    selected_label: str,
    selected_credential: ProviderCredential,
    program: TypedRelationalProgram,
    transport: ModelTransport,
    all_start_barrier: threading.Barrier,
    model_concurrency: ModelConcurrency,
) -> UnitOutcome:
    all_start_barrier.wait(timeout=60)
    _claim_path, terminal_path = _work_paths(output_root, work)
    model_call_count = 0
    operator_call_count = 0
    model_response_received = False
    response_hash: str | None = None
    output: dict[str, Any] | None = None
    error_code: str | None = None
    try:
        if work.arm in {"raw_model", "agent_typed_model"}:
            if work.model_request is None:
                raise DevelopmentRunnerError("model work lacks its request")
            with model_concurrency.acquire():
                model_call_count = 1
                content = transport.complete(
                    credential=selected_credential,
                    request=work.model_request,
                )
            model_response_received = True
            response_hash = payload_hash({"content": content})
            if work.arm == "raw_model":
                output = _parse_raw_response(content)
            else:
                _parse_agent_response(
                    content, program_id=str(work.unit["program_id"])
                )
                operator_call_count = 1
                output = _parse_operator_output(
                    execute_frozen_operator(program, work.input_path)
                )
        else:
            operator_call_count = 1
            output = _parse_operator_output(
                execute_frozen_operator(program, work.input_path)
            )
    except Exception as exc:
        error_code = _terminal_error_code(exc, arm=work.arm)
        output = None
    body = {
        "arm": work.arm,
        "attempt_count": 1,
        "claim_hash": claim["claim_hash"],
        "error_code": error_code,
        "execution_terminal": True,
        "model_call_count": model_call_count,
        "model_response_hash": response_hash,
        "model_response_received": model_response_received,
        "operator_call_count": operator_call_count,
        "output": output,
        "output_contract_valid": output is not None,
        "output_hash": payload_hash(output) if output is not None else None,
        "prompt_persisted": False,
        "provider_label_hash": (
            payload_hash({"provider_label": selected_label})
            if work.arm in {"raw_model", "agent_typed_model"}
            else None
        ),
        "raw_response_persisted": False,
        "secret_value_persisted": False,
        "terminal_version": WORK_TERMINAL_VERSION,
        "trace_persisted": False,
        "work_unit_hash": work.unit_hash,
        "work_unit_id": work.work_unit_id,
    }
    terminal = atomic_write_hashed_json_v2(
        terminal_path, body, hash_field="terminal_hash"
    )
    return UnitOutcome(prepared=work, terminal=terminal, recovered=False)


def _safe_failure_receipt(
    output_root: Path, exc: Exception, *, oracle_released: bool
) -> None:
    body = {
        "development_gold_persisted": False,
        "error_message_hash": payload_hash({"message": str(exc)}),
        "error_type": type(exc).__name__,
        "execution_completed": False,
        "model_replay_authorized": False,
        "offline_oracle_released": oracle_released,
        "prompt_persisted": False,
        "raw_error_persisted": False,
        "runner_version": RUNNER_VERSION,
        "secret_value_persisted": False,
        "trace_persisted": False,
    }
    attempt_hash = payload_hash(body)
    try:
        _write_or_verify_receipt(
            output_root / "failures" / f"{attempt_hash}.json",
            body,
            hash_field="failure_hash",
        )
    except Exception:
        pass


def _verify_execution_inputs(
    *,
    development_root: Path,
    public_freeze: Mapping[str, Any],
    provider_identity_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], TypedRelationalProgram]:
    worker = verify_worker_plan(
        read_json(development_root / WORKER_PLAN_FILENAME),
        development_root=development_root,
    )
    controller = verify_controller_plan(
        read_json(development_root / CONTROLLER_PLAN_FILENAME),
        worker_plan=worker,
    )
    bindings = public_freeze["binding_hashes"]
    if (
        controller.get("development_root") != str(development_root)
        or controller.get("development_root_commitment")
        != payload_hash(str(development_root))
        or controller.get("development_root_commitment")
        != bindings.get("development_root_commitment")
        or controller.get("controller_plan_hash")
        != bindings.get("controller_plan_hash")
        or worker.get("worker_plan_hash") != bindings.get("worker_plan_hash")
        or bindings.get("development_schema_set_hash")
        != DEVELOPMENT_SCHEMA_SET_HASH
        or bindings.get("provider_identity_hash") != provider_identity_hash
        or bindings.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
    ):
        raise DevelopmentRunnerError("development execution bindings drifted")
    source_binding = controller["source_view_binding"]
    expected_source_bindings = {
        "development_source_index_file_sha256": bindings[
            "development_source_index_file_sha256"
        ],
        "development_source_index_hash": bindings[
            "development_source_index_hash"
        ],
        "development_source_receipt_file_sha256": bindings[
            "development_source_receipt_file_sha256"
        ],
        "development_source_receipt_hash": bindings[
            "development_source_receipt_hash"
        ],
        "source_view_input_set_hash": bindings[
            "development_source_input_set_hash"
        ],
        "source_view_tree_hash": bindings["development_source_tree_hash"],
        "staged_input_set_hash": bindings["staged_input_set_hash"],
    }
    if source_binding != expected_source_bindings:
        raise DevelopmentRunnerError(
            "controller/public source-view binding drifted"
        )
    operator_binding = worker["operator_binding"]
    operator_path = development_root / str(
        operator_binding["frozen_program_relative_path"]
    )
    if operator_path.is_symlink() or not operator_path.is_file():
        raise DevelopmentRunnerError("frozen operator is not a regular file")
    program = load_frozen_program(operator_path)
    envelope = read_json(operator_path)
    if (
        program.program_hash != operator_binding.get("program_id")
        or program.program_hash != bindings.get("candidate_program_id")
        or operator_binding.get("operator_version") != OPERATOR_VERSION
        or payload_hash(envelope) != operator_binding.get("program_envelope_hash")
        or sha256_file(operator_path)
        != operator_binding.get("frozen_program_file_sha256")
        or operator_binding.get("frozen_program_file_sha256")
        != bindings.get("candidate_program_file_sha256")
    ):
        raise DevelopmentRunnerError("frozen operator binding drifted")
    return worker, controller, program


def _prepare_work_units(
    *, development_root: Path, worker: Mapping[str, Any]
) -> tuple[PreparedWork, ...]:
    items = {
        str(item["anonymous_item_id"]): item for item in worker["items"]
    }
    prepared: list[PreparedWork] = []
    for unit in worker["work_units"]:
        item = items[str(unit["anonymous_item_id"])]
        input_path = development_root / str(item["input_relative_path"])
        if input_path.is_symlink() or not input_path.is_file():
            raise DevelopmentRunnerError("staged development input is not regular")
        if sha256_file(input_path) != unit.get("input_sha256"):
            raise DevelopmentRunnerError("staged development input drifted")
        try:
            csv_text = input_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DevelopmentRunnerError("staged development input is unreadable") from exc
        arm = str(unit["arm"])
        model_request: ModelRequest | None = None
        if arm == "raw_model":
            model_request = ModelRequest(
                purpose=f"raw_model:{unit['work_unit_id']}",
                system_prompt=RAW_SYSTEM_PROMPT,
                user_payload={
                    "input_csv_utf8": csv_text,
                    "output_contract": TASK_CONTRACT["output"],
                    "request_kind": "noaa_gsod_raw_task_v1",
                    "task_contract": TASK_CONTRACT,
                },
            )
        elif arm == "agent_typed_model":
            model_request = ModelRequest(
                purpose=f"agent_typed_model:{unit['work_unit_id']}",
                system_prompt=AGENT_SYSTEM_PROMPT,
                user_payload={
                    "input_csv_utf8": csv_text,
                    "output_schema": {
                        "additionalProperties": False,
                        "properties": {
                            "program_id": {"enum": [unit["program_id"]]}
                        },
                        "required": ["program_id"],
                        "type": "object",
                    },
                    "request_kind": "noaa_gsod_typed_operator_selection_v1",
                    "task_contract": TASK_CONTRACT,
                },
            )
        if (
            model_request is not None
            and model_request.canonical_body_byte_count
            > MODEL_REQUEST_BODY_BYTE_BUDGET
        ):
            raise DevelopmentRunnerError(
                "complete model request body exceeds the frozen byte budget"
            )
        prepared.append(
            PreparedWork(
                unit=dict(unit),
                unit_hash=payload_hash(unit),
                input_path=input_path,
                model_request=model_request,
            )
        )
    if (
        len(prepared) != WORK_UNIT_COUNT
        or len({row.work_unit_id for row in prepared}) != WORK_UNIT_COUNT
        or {row.arm for row in prepared} != set(ARM_IDS)
    ):
        raise DevelopmentRunnerError("prepared work grid drifted")
    return tuple(sorted(prepared, key=lambda row: row.work_unit_id))


def _launch_transition(
    *,
    output_root: Path,
    public_freeze: Mapping[str, Any],
    worker: Mapping[str, Any],
    controller: Mapping[str, Any],
    selection: Mapping[str, Any],
    prepared: Sequence[PreparedWork],
) -> dict[str, Any]:
    model_request_hashes = {
        work.work_unit_id: work.model_request.request_hash
        for work in prepared
        if work.model_request is not None
    }
    if len(model_request_hashes) != MODEL_WORK_UNIT_COUNT:
        raise DevelopmentRunnerError(
            "launch does not precommit the exact 12 model requests"
        )
    body = {
        "attempts_per_work_unit": 1,
        "controller_plan_hash": controller["controller_plan_hash"],
        "launch_authorized": True,
        "launch_transition_version": LAUNCH_TRANSITION_VERSION,
        "maximum_model_concurrency": MODEL_WORK_UNIT_COUNT,
        "mid_batch_provider_switch_authorized": False,
        "model_request_hash_count": MODEL_WORK_UNIT_COUNT,
        "model_request_hash_set_hash": payload_hash(model_request_hashes),
        "model_request_hashes": model_request_hashes,
        "model_request_hashes_precommitted": True,
        "performance_gate_applied": False,
        "pre_run_freeze_hash": public_freeze["pre_run_freeze_hash"],
        "promotion_authorized": False,
        "replay_authorized": False,
        "resampling_authorized": False,
        "retry_authorized": False,
        "selection_hash": selection["selection_hash"],
        "total_work_units": WORK_UNIT_COUNT,
        "worker_plan_hash": worker["worker_plan_hash"],
    }
    return _write_or_verify_receipt(
        output_root / LAUNCH_FILENAME, body, hash_field="launch_hash"
    )


OracleFunction = Callable[[str | Path], dict[str, Any]]


def _offline_evaluate(
    *,
    output_root: Path,
    prepared: Sequence[PreparedWork],
    outcomes: Sequence[UnitOutcome],
    oracle_functions: Sequence[tuple[str, OracleFunction]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(outcomes) != WORK_UNIT_COUNT or any(
        outcome.terminal is None or outcome.blocked_incomplete
        for outcome in outcomes
    ):
        raise DevelopmentRunnerError(
            "offline oracle cannot release before all terminal joins"
        )
    if tuple(identifier for identifier, _function in oracle_functions) != ORACLE_IDS:
        raise DevelopmentRunnerError("offline oracle identity drifted")
    work_by_item: dict[str, dict[str, UnitOutcome]] = {}
    input_by_item: dict[str, Path] = {}
    for outcome in outcomes:
        item_id = outcome.prepared.anonymous_item_id
        work_by_item.setdefault(item_id, {})[outcome.prepared.arm] = outcome
        input_by_item[item_id] = outcome.prepared.input_path
    rows: list[dict[str, Any]] = []
    arm_successes = {arm: 0 for arm in ARM_IDS}
    arm_valid = {arm: 0 for arm in ARM_IDS}
    pair_specs = {
        "agent_typed_minus_operator_only": (
            "agent_typed_model",
            "operator_only_local",
        ),
        "agent_typed_minus_raw": ("agent_typed_model", "raw_model"),
        "operator_only_minus_raw": ("operator_only_local", "raw_model"),
    }
    paired = {
        name: {
            "complete_pair_count": 0,
            "gain_count": 0,
            "harm_count": 0,
            "incomplete_pair_count": 0,
            "tie_count": 0,
        }
        for name in pair_specs
    }
    oracle_calls = 0
    oracle_failures = 0
    disagreements = 0
    for item_id in sorted(work_by_item):
        oracle_outputs: list[dict[str, Any] | None] = []
        oracle_hashes: dict[str, str | None] = {}
        oracle_errors: dict[str, str | None] = {}
        for identifier, function in oracle_functions:
            oracle_calls += 1
            try:
                value = _validate_task_output(function(input_by_item[item_id]))
                oracle_outputs.append(value)
                oracle_hashes[identifier] = payload_hash(value)
                oracle_errors[identifier] = None
            except Exception as exc:
                oracle_outputs.append(None)
                oracle_hashes[identifier] = None
                oracle_errors[identifier] = type(exc).__name__
                oracle_failures += 1
        consensus = (
            oracle_outputs[0]
            if all(value is not None for value in oracle_outputs)
            and oracle_outputs[0] == oracle_outputs[1]
            else None
        )
        disagreement = (
            all(value is not None for value in oracle_outputs)
            and oracle_outputs[0] != oracle_outputs[1]
        )
        disagreements += int(disagreement)
        arms: dict[str, Any] = {}
        for arm in ARM_IDS:
            terminal = work_by_item[item_id][arm].terminal
            assert terminal is not None
            valid = terminal.get("output_contract_valid") is True
            exact = valid and consensus is not None and terminal.get("output") == consensus
            arm_valid[arm] += int(valid)
            arm_successes[arm] += int(exact)
            arms[arm] = {
                "exact": exact,
                "output_contract_valid": valid,
                "output_hash": terminal.get("output_hash"),
                "terminal_hash": terminal.get("terminal_hash"),
            }
        for name, (first_arm, second_arm) in pair_specs.items():
            first = arms[first_arm]
            second = arms[second_arm]
            # Intention-to-treat: once both assigned work units have terminal
            # outcomes, contract-invalid or transport-failed outputs are
            # observed incorrect outcomes, not missing observations.  Only an
            # unavailable offline consensus makes the pair incomplete.
            complete = consensus is not None
            if not complete:
                paired[name]["incomplete_pair_count"] += 1
            else:
                paired[name]["complete_pair_count"] += 1
                first_exact = first["exact"] is True
                second_exact = second["exact"] is True
                if first_exact and not second_exact:
                    paired[name]["gain_count"] += 1
                elif second_exact and not first_exact:
                    paired[name]["harm_count"] += 1
                else:
                    paired[name]["tie_count"] += 1
        rows.append(
            {
                "anonymous_item_id": item_id,
                "arms": arms,
                "oracle_disagreement": disagreement,
                "oracle_errors": oracle_errors,
                "oracle_output_hashes": oracle_hashes,
            }
        )
    private_body = {
        "development_gold_persisted": False,
        "evaluation_version": PRIVATE_EVALUATION_VERSION,
        "item_count": len(rows),
        "oracle_call_count": oracle_calls,
        "oracle_ids": list(ORACLE_IDS),
        "oracle_release_after_join_count": WORK_UNIT_COUNT,
        "rows": rows,
        "rows_hash": payload_hash(rows),
    }
    private_receipt = _write_or_verify_receipt(
        output_root / PRIVATE_EVALUATION_FILENAME,
        private_body,
        hash_field="evaluation_hash",
    )
    for counts in paired.values():
        counts["paired_net_gain"] = (
            counts["gain_count"] - counts["harm_count"]
        )
        if (
            counts["complete_pair_count"]
            != counts["gain_count"]
            + counts["harm_count"]
            + counts["tie_count"]
            or counts["complete_pair_count"]
            + counts["incomplete_pair_count"]
            != len(rows)
        ):
            raise DevelopmentRunnerError("item-paired accounting drifted")
    pairwise_net_deltas = {
        "agent_typed_minus_raw": (
            arm_successes["agent_typed_model"]
            - arm_successes["raw_model"]
        ),
        "operator_only_minus_raw": (
            arm_successes["operator_only_local"]
            - arm_successes["raw_model"]
        ),
        "agent_typed_minus_operator_only": (
            arm_successes["agent_typed_model"]
            - arm_successes["operator_only_local"]
        ),
    }
    for name, delta in pairwise_net_deltas.items():
        if paired[name]["paired_net_gain"] != delta:
            raise DevelopmentRunnerError(
                "ITT paired net gain differs from the arm success delta"
            )
    aggregate = {
        "arm_contract_valid_counts": arm_valid,
        "arm_exact_success_counts": arm_successes,
        "oracle_call_count": oracle_calls,
        "oracle_disagreement_count": disagreements,
        "oracle_failure_count": oracle_failures,
        "paired_estimand": "intention_to_treat_terminal_failure_as_incorrect_v1",
        "pairwise_item_counts": paired,
        "pairwise_net_deltas": pairwise_net_deltas,
    }
    return aggregate, private_receipt


def _validate_safe_report(
    report_body: Mapping[str, Any], *, credentials: Sequence[ProviderCredential]
) -> None:
    serialized = canonical_json_bytes(report_body)
    forbidden = [
        credential.api_key.encode("utf-8") for credential in credentials
    ]
    forbidden.extend(
        [
            b"input_csv_utf8",
            b"oracle_outputs",
            b"gold_commitment",
            b"system_prompt",
            b"model_response_hash",
            b'"output"',
            b'"prompt"',
            b'"trace"',
        ]
    )
    if any(token and token in serialized for token in forbidden):
        raise DevelopmentRunnerError("public report contains forbidden content")


def _run_development_core(
    *,
    development_root: str | Path,
    public_freeze_path: str | Path,
    output_root: str | Path,
    plus_env_file: str | Path,
    pro_env_file: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
    transport: ModelTransport,
    oracle_functions: Sequence[tuple[str, OracleFunction]],
    dependency_injection_used: bool,
) -> dict[str, Any]:
    formal_evidence = not dependency_injection_used
    execution_dependency_ids = _execution_dependency_ids(
        dependency_injection_used=dependency_injection_used
    )
    if not dependency_injection_used and (
        type(transport) is not UrllibOpenAICompatibleTransport
        or tuple(oracle_functions)
        != (
            (oracle_stdlib.ORACLE_ID, oracle_stdlib.evaluate),
            (oracle_sqlite.ORACLE_ID, oracle_sqlite.evaluate),
        )
    ):
        raise DevelopmentRunnerError(
            "formal execution dependencies are not the exact built-in implementations"
        )
    public_freeze = verify_public_pre_run_freeze(
        read_json(Path(public_freeze_path).resolve(strict=True))
    )
    development = Path(development_root).expanduser().resolve(strict=True)
    if development.is_symlink() or not development.is_dir():
        raise DevelopmentRunnerError("development root is not a regular directory")
    destination_request = Path(output_root).expanduser()
    if destination_request.is_symlink():
        raise DevelopmentRunnerError("output root is symbolic")
    destination = destination_request.resolve()
    if destination.parent != development:
        raise DevelopmentRunnerError(
            "formal output must be one direct child of the development root"
        )

    # Compare the current executable source to the prospective freeze before
    # credentials are opened, durable launch state is created, or any provider
    # canary/task request is sent. Historical freeze verification remains
    # structural; this is the execution-time live-byte comparison.
    live_implementation_set = build_development_implementation_set()
    if live_implementation_set["implementation_set_hash"] != (
        public_freeze["binding_hashes"].get("implementation_set_hash")
    ):
        raise DevelopmentRunnerError(
            "live development implementation differs from pre-run freeze"
        )

    # Credentials and provider route are resolved before worker/controller plans
    # or development inputs are opened.
    plus = load_provider_credential(plus_env_file, channel_id=plus_channel_id)
    pro = load_provider_credential(pro_env_file, channel_id=pro_channel_id)
    provider_identity = ProviderIdentity(
        plus_channel_id=plus.channel_id,
        plus_endpoint_origin=plus.api_origin,
        pro_channel_id=pro.channel_id,
        pro_endpoint_origin=pro.api_origin,
    )
    provider_identity.validate()
    if (
        provider_identity.identity_hash
        != public_freeze["binding_hashes"].get("provider_identity_hash")
    ):
        raise DevelopmentRunnerError("provider identity differs from public freeze")
    if destination.exists() and not destination.is_dir():
        raise DevelopmentRunnerError("formal output root is not a directory")
    destination.mkdir(parents=False, exist_ok=True)
    if destination.is_symlink():
        raise DevelopmentRunnerError("formal output root became symbolic")
    selected_transport = transport
    oracle_released = False
    try:
        precommit = _provider_precommit(
            output_root=destination,
            public_freeze=public_freeze,
            plus=plus,
            pro=pro,
            dependency_injection_used=dependency_injection_used,
            execution_dependency_ids=execution_dependency_ids,
            formal_evidence=formal_evidence,
        )
        if _has_any_work_state(destination) and not (
            destination / PROVIDER_SELECTION_FILENAME
        ).exists():
            raise NoReplayError(
                "work state exists without a completed provider selection; "
                "canary/task replay forbidden"
            )
        selected_label, selected_credential, selection = select_provider(
            output_root=destination,
            precommit=precommit,
            plus=plus,
            pro=pro,
            transport=selected_transport,
        )
        worker, controller, program = _verify_execution_inputs(
            development_root=development,
            public_freeze=public_freeze,
            provider_identity_hash=provider_identity.identity_hash,
        )
        prepared = _prepare_work_units(
            development_root=development, worker=worker
        )
        selection_hash = str(selection["selection_hash"])
        grid_state, recovered_outcomes = _inspect_existing_work_grid(
            output_root=destination,
            prepared=prepared,
            selection_hash=selection_hash,
        )
        launch_path = destination / LAUNCH_FILENAME
        report_path = destination / REPORT_FILENAME
        if grid_state == "fresh":
            if (
                launch_path.exists()
                or launch_path.is_symlink()
                or report_path.exists()
                or report_path.is_symlink()
                or (destination / WORK_CLAIM_SET_FILENAME).exists()
                or (destination / WORK_CLAIM_SET_FILENAME).is_symlink()
            ):
                raise NoReplayError(
                    "fresh worker grid conflicts with consumed batch state"
                )
            launch = _launch_transition(
                output_root=destination,
                public_freeze=public_freeze,
                worker=worker,
                controller=controller,
                selection=selection,
                prepared=prepared,
            )
            claims = _preclaim_complete_work_grid(
                output_root=destination,
                prepared=prepared,
                selection_hash=selection_hash,
            )
            all_start_barrier = threading.Barrier(WORK_UNIT_COUNT)
            model_concurrency = ModelConcurrency(
                slots=MODEL_WORK_UNIT_COUNT,
                start_participants=MODEL_WORK_UNIT_COUNT,
            )

            def run_one(work: PreparedWork) -> UnitOutcome:
                return _run_work_once(
                    output_root=destination,
                    work=work,
                    claim=claims[work.work_unit_id],
                    selected_label=selected_label,
                    selected_credential=selected_credential,
                    program=program,
                    transport=selected_transport,
                    all_start_barrier=all_start_barrier,
                    model_concurrency=model_concurrency,
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=WORK_UNIT_COUNT
            ) as executor:
                futures = tuple(
                    executor.submit(run_one, work) for work in prepared
                )
                # All 18 durable claims and all 18 futures exist before any
                # future is read. A failed work unit is never retried.
                outcomes = tuple(future.result() for future in futures)
        else:
            if not launch_path.exists() or launch_path.is_symlink():
                raise NoReplayError(
                    "complete terminals lack the prior atomic launch transition"
                )
            launch = _launch_transition(
                output_root=destination,
                public_freeze=public_freeze,
                worker=worker,
                controller=controller,
                selection=selection,
                prepared=prepared,
            )
            _validate_complete_work_claim_set(
                output_root=destination,
                prepared=prepared,
                selection_hash=selection_hash,
            )
            outcomes = recovered_outcomes
            model_concurrency = ModelConcurrency(
                slots=MODEL_WORK_UNIT_COUNT,
                start_participants=0,
            )
            if report_path.exists() or report_path.is_symlink():
                if report_path.is_symlink():
                    raise DevelopmentRunnerError("development report is symbolic")
                completed_report = read_hashed_json_v2(
                    report_path, hash_field="report_hash"
                )
                if (
                    completed_report.get("execution_completed") is not True
                    or completed_report.get("joined_work_unit_count")
                    != WORK_UNIT_COUNT
                    or completed_report.get("pre_run_freeze_hash")
                    != public_freeze["pre_run_freeze_hash"]
                    or completed_report.get("selection_hash") != selection_hash
                    or completed_report.get("launch_hash")
                    != launch["launch_hash"]
                    or completed_report.get("worker_plan_hash")
                    != worker["worker_plan_hash"]
                    or completed_report.get("dependency_injection_used")
                    is not dependency_injection_used
                    or completed_report.get("formal_evidence")
                    is not formal_evidence
                    or completed_report.get("execution_dependency_ids")
                    != execution_dependency_ids
                ):
                    raise DevelopmentRunnerError(
                        "completed development report binding drifted"
                    )
                _validate_safe_report(
                    completed_report, credentials=(plus, pro)
                )
                return completed_report
        if len(outcomes) != WORK_UNIT_COUNT or any(
            outcome.blocked_incomplete or outcome.terminal is None
            for outcome in outcomes
        ):
            raise NoReplayError(
                "the complete 18-unit grid did not reach terminal join"
            )

        oracle_released = True
        aggregate, private_evaluation = _offline_evaluate(
            output_root=destination,
            prepared=prepared,
            outcomes=outcomes,
            oracle_functions=oracle_functions,
        )
        task_model_calls = sum(
            int(outcome.terminal["model_call_count"])
            for outcome in outcomes
            if outcome.terminal is not None
        )
        operator_calls = sum(
            int(outcome.terminal["operator_call_count"])
            for outcome in outcomes
            if outcome.terminal is not None
        )
        canary_model_calls = 1 + int(selected_label == "pro")
        recovered_count = sum(outcome.recovered for outcome in outcomes)
        execution_integrity_valid = (
            task_model_calls == MODEL_WORK_UNIT_COUNT
            and len(outcomes) == WORK_UNIT_COUNT
            and all(
                outcome.terminal is not None
                and outcome.terminal.get("execution_terminal") is True
                for outcome in outcomes
            )
        )
        paired_evidence_complete = (
            aggregate["oracle_disagreement_count"] == 0
            and aggregate["oracle_failure_count"] == 0
            and all(
                counts["complete_pair_count"] == DEVELOPMENT_ITEM_COUNT
                and counts["incomplete_pair_count"] == 0
                for counts in aggregate["pairwise_item_counts"].values()
            )
        )
        formal_evidence_valid = (
            formal_evidence
            and execution_integrity_valid
            and paired_evidence_complete
        )
        body = {
            "call_ledger": {
                "canary_model_calls": canary_model_calls,
                "offline_oracle_calls": aggregate["oracle_call_count"],
                "online_judge_calls": 0,
                "operator_calls": operator_calls,
                "replays": 0,
                "resamples": 0,
                "retries": 0,
                "scoring_model_calls": 0,
                "task_model_calls": task_model_calls,
                "total_model_calls": canary_model_calls + task_model_calls,
            },
            "concurrency": {
                "all_claims_persisted_before_work_start": True,
                "all_futures_submitted_before_results_read": (
                    grid_state == "fresh"
                ),
                "configured_model_concurrency": MODEL_WORK_UNIT_COUNT,
                "configured_work_concurrency": WORK_UNIT_COUNT,
                "observed_maximum_model_calls": model_concurrency.maximum_active,
                "recovered_from_complete_terminal_grid": (
                    grid_state == "complete_terminals"
                ),
                "single_task_batch": True,
            },
            "content_boundary": {
                "development_gold_persisted_publicly": False,
                "development_raw_input_persisted_publicly": False,
                "model_answer_persisted_publicly": False,
                "prompt_persisted_publicly": False,
                "secret_value_persisted": False,
                "trace_persisted_publicly": False,
            },
            "controller_plan_hash": controller["controller_plan_hash"],
            "dependency_injection_used": dependency_injection_used,
            "descriptive_only": True,
            "development_consumed": True,
            "evidence_valid": formal_evidence_valid,
            "execution_dependency_ids": execution_dependency_ids,
            "execution_integrity_valid": execution_integrity_valid,
            "execution_completed": True,
            "formal_evidence": formal_evidence,
            "formal_evidence_valid": formal_evidence_valid,
            "joined_work_unit_count": len(outcomes),
            "launch_hash": launch["launch_hash"],
            "offline_evaluation": aggregate,
            "offline_evaluation_only": True,
            "paired_evidence_complete": paired_evidence_complete,
            "performance_gate_applied": False,
            "pre_run_freeze_hash": public_freeze["pre_run_freeze_hash"],
            "private_evaluation_hash": private_evaluation["evaluation_hash"],
            "promotion_authorized": False,
            "provider_identity_precommit_hash": precommit["precommit_hash"],
            "recovered_work_unit_count": recovered_count,
            "runner_version": RUNNER_VERSION,
            "sealed_content_accessed": False,
            "selected_provider_fixed_for_complete_batch": True,
            "selected_provider_label": selected_label,
            "selection_hash": selection["selection_hash"],
            "study_id": STUDY_ID,
            "worker_plan_hash": worker["worker_plan_hash"],
        }
        _validate_safe_report(body, credentials=(plus, pro))
        return _write_or_verify_receipt(
            destination / REPORT_FILENAME, body, hash_field="report_hash"
        )
    except Exception as exc:
        _safe_failure_receipt(
            destination, exc, oracle_released=oracle_released
        )
        raise


def run_formal_development(
    *,
    development_root: str | Path,
    public_freeze_path: str | Path,
    output_root: str | Path,
    plus_env_file: str | Path,
    pro_env_file: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
) -> dict[str, Any]:
    """Run formal evidence with only the frozen built-in dependencies."""

    return _run_development_core(
        development_root=development_root,
        public_freeze_path=public_freeze_path,
        output_root=output_root,
        plus_env_file=plus_env_file,
        pro_env_file=pro_env_file,
        plus_channel_id=plus_channel_id,
        pro_channel_id=pro_channel_id,
        transport=UrllibOpenAICompatibleTransport(),
        oracle_functions=(
            (oracle_stdlib.ORACLE_ID, oracle_stdlib.evaluate),
            (oracle_sqlite.ORACLE_ID, oracle_sqlite.evaluate),
        ),
        dependency_injection_used=False,
    )


def run_synthetic_development_for_tests(
    *,
    development_root: str | Path,
    public_freeze_path: str | Path,
    output_root: str | Path,
    plus_env_file: str | Path,
    pro_env_file: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
    transport: ModelTransport,
    oracle_functions: Sequence[tuple[str, OracleFunction]] | None = None,
) -> dict[str, Any]:
    """Exercise the runner with injected dependencies without formal evidence."""

    selected_oracles = oracle_functions or (
        (oracle_stdlib.ORACLE_ID, oracle_stdlib.evaluate),
        (oracle_sqlite.ORACLE_ID, oracle_sqlite.evaluate),
    )
    return _run_development_core(
        development_root=development_root,
        public_freeze_path=public_freeze_path,
        output_root=output_root,
        plus_env_file=plus_env_file,
        pro_env_file=pro_env_file,
        plus_channel_id=plus_channel_id,
        pro_channel_id=pro_channel_id,
        transport=transport,
        oracle_functions=selected_oracles,
        dependency_injection_used=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--public-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plus-env", required=True)
    parser.add_argument("--pro-env", required=True)
    parser.add_argument("--plus-channel-id", required=True)
    parser.add_argument("--pro-channel-id", required=True)
    arguments = parser.parse_args(argv)
    report = run_formal_development(
        development_root=arguments.development_root,
        public_freeze_path=arguments.public_freeze,
        output_root=arguments.output_root,
        plus_env_file=arguments.plus_env,
        pro_env_file=arguments.pro_env,
        plus_channel_id=arguments.plus_channel_id,
        pro_channel_id=arguments.pro_channel_id,
    )
    print(
        json.dumps(
            {
                "execution_completed": report["execution_completed"],
                "execution_integrity_valid": report[
                    "execution_integrity_valid"
                ],
                "formal_evidence_valid": report["formal_evidence_valid"],
                "paired_evidence_complete": report[
                    "paired_evidence_complete"
                ],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DevelopmentRunnerError",
    "ModelRequest",
    "ModelTransport",
    "ProviderCredential",
    "ProviderProtocolError",
    "ProviderTransportUnavailable",
    "RUNNER_VERSION",
    "UrllibOpenAICompatibleTransport",
    "api_key_hmac_commitment",
    "endpoint_identity_hash",
    "load_provider_credential",
    "main",
    "run_formal_development",
    "run_synthetic_development_for_tests",
    "select_provider",
]
