from __future__ import annotations

"""Finite provider identity binding for the fresh SEC-13F batch."""

import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from assumption_agent.events import Event
from assumption_agent.models import stable_hash
from assumption_agent.secure_env import (
    LOCKED_MODEL,
    load_dotenv,
    map_legacy_model_env,
)
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
)


PROVIDER_IDENTITY_SIDECAR_VERSION = (
    "financial_sec13f_contract_provider_identity_sidecar_v1"
)
EXECUTION_PROVIDER_BINDING_VERSION = (
    "financial_sec13f_contract_execution_provider_binding_v1"
)
API_KEY_COMMITMENT_VERSION = (
    "financial_sec13f_contract_api_key_hmac_sha256_v1"
)
API_KEY_COMMITMENT_CHALLENGE = (
    b"financial-sec13f-contract/provider-key/pre-batch/v1"
)
PROVIDER_SELECTION_POLICY = (
    "pretask_plus_model_response_then_unavailable_only_pro_single_batch_v3"
)
FALLBACK_POLICY = "pro_only_after_complete_plus_unavailability"
PROPOSAL_CANARY_VERSION = "proposal_canary_v1"
REGISTERED_PROVIDER_LABELS = frozenset({"plus", "pro"})

_CANONICAL_API_KEY = "ASSUMPTION_V2_API_KEY"
_CANONICAL_API_BASE = "ASSUMPTION_V2_API_BASE"
_KEY_SOURCE_ORDER = (
    _CANONICAL_API_KEY,
    "RUOLI_GPT_KEY",
    "GPT5_API_KEY",
)
_BASE_SOURCE_ORDER = (
    _CANONICAL_API_BASE,
    "RUOLI_BASE_URL",
    "GPT5_BASE_URL",
)
PROVIDER_ENVIRONMENT_KEYS = frozenset(
    {
        *_KEY_SOURCE_ORDER,
        *_BASE_SOURCE_ORDER,
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ASSUMPTION_V2_MODEL",
        "ASSUMPTION_V2_ALLOW_ALTERNATE_MODEL",
        "ASSUMPTION_V2_PROVIDER_CHAIN",
        "ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE",
        "ASSUMPTION_V2_API_ALLOWED_IPV4S",
    }
)

_SIDECAR_FIELDS = {
    "sidecar_version",
    "provider_label",
    "provider_label_hash",
    "proposal_canary_version",
    "canary_payload_hash",
    "canary_report_file_sha256",
    "event_ledger_hash",
    "event_ledger_file_sha256",
    "event_count",
    "model",
    "api_origin",
    "api_key_commitment_version",
    "api_key_commitment_challenge_sha256",
    "api_key_hmac_sha256",
    "model_response_received",
    "semantic_acceptance_used_for_provider_selection",
    "environment_aliases_scrubbed_before_canary",
    "single_env_file_loaded",
    "env_file_content_persisted",
    "api_key_persisted",
    "secret_value_persisted",
    "raw_canary_content_persisted",
    "sidecar_hash",
}

_BINDING_FIELDS = {
    "binding_version",
    "provider_label",
    "provider_label_hash",
    "fallback_policy",
    "identity_sidecar_relative_path",
    "identity_sidecar_file_sha256",
    "identity_sidecar_hash",
    "selected_canary_relative_path",
    "selected_canary_file_sha256",
    "selected_event_ledger_relative_path",
    "selected_event_ledger_file_sha256",
    "selection_receipt_relative_path",
    "selection_receipt_file_sha256",
    "selection_receipt_hash",
    "model",
    "api_origin",
    "api_key_commitment_version",
    "api_key_hmac_sha256",
    "plus_transport_failure_before_pro_selection",
    "selected_provider_fixed_for_complete_batch",
    "mid_batch_provider_switch_authorized",
    "mid_batch_retry_authorized",
    "secret_value_persisted",
    "binding_hash",
}


class ProviderIdentityError(RuntimeError):
    """A provider route or key no longer matches its pre-batch identity."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ProviderIdentityError(f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderIdentityError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ProviderIdentityError(f"{label} must contain one object")
    return value, raw


def _normalize_origin(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        port_value = parsed.port
    except ValueError as exc:
        raise ProviderIdentityError("provider API origin is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderIdentityError("provider API origin is malformed")
    port = f":{port_value}" if port_value is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _dotenv_identity_values(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ProviderIdentityError("provider env file is not regular")
    if source.stat().st_size > 1024 * 1024:
        raise ProviderIdentityError("provider env file exceeds its byte bound")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProviderIdentityError("provider env file is unreadable") from exc
    values: dict[str, str] = {}
    controlled_seen: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not key:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if key in PROVIDER_ENVIRONMENT_KEYS:
            if key in controlled_seen:
                raise ProviderIdentityError(
                    "provider env contains a duplicate identity alias"
                )
            controlled_seen.add(key)
            values[key] = value
    return values


def _first_nonempty(
    values: Mapping[str, str], names: Sequence[str]
) -> str:
    for name in names:
        value = str(values.get(name) or "").strip()
        if value:
            return value
    return ""


def api_key_hmac_commitment_v1(api_key: str) -> str:
    normalized = str(api_key).strip()
    if not normalized:
        raise ProviderIdentityError("provider API key is absent")
    return hmac.new(
        normalized.encode("utf-8"),
        API_KEY_COMMITMENT_CHALLENGE,
        hashlib.sha256,
    ).hexdigest()


def provider_environment_identity_from_file_v1(
    env_file: str | Path,
) -> dict[str, Any]:
    values = _dotenv_identity_values(env_file)
    api_key = _first_nonempty(values, _KEY_SOURCE_ORDER)
    api_base = _first_nonempty(values, _BASE_SOURCE_ORDER)
    if not api_key or not api_base:
        raise ProviderIdentityError(
            "provider env lacks one effective API key/base identity"
        )
    model = str(values.get("ASSUMPTION_V2_MODEL") or LOCKED_MODEL).strip()
    if not model:
        raise ProviderIdentityError("provider env model is absent")
    return {
        "model": model,
        "api_origin": _normalize_origin(api_base),
        "api_key_commitment_version": API_KEY_COMMITMENT_VERSION,
        "api_key_hmac_sha256": api_key_hmac_commitment_v1(api_key),
        "env_file_content_persisted": False,
        "api_key_persisted": False,
        "secret_value_persisted": False,
    }


def scrub_provider_environment_v1(
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    target = os.environ if environment is None else environment
    for name in PROVIDER_ENVIRONMENT_KEYS:
        target.pop(name, None)
    return target


def load_provider_environment_v1(env_file: str | Path) -> dict[str, Any]:
    """Replace ambient provider aliases from exactly one dotenv file."""

    expected = provider_environment_identity_from_file_v1(env_file)
    scrub_provider_environment_v1()
    load_dotenv(env_file, override=True)
    map_legacy_model_env(override=False)
    # Remove secondary key/base aliases after canonicalization.  Downstream
    # code receives exactly one effective API identity.
    for name in (
        "RUOLI_GPT_KEY",
        "GPT5_API_KEY",
        "RUOLI_BASE_URL",
        "GPT5_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        os.environ.pop(name, None)
    observed_key = os.environ.get(_CANONICAL_API_KEY, "").strip()
    observed_base = os.environ.get(_CANONICAL_API_BASE, "").strip()
    observed_model = os.environ.get("ASSUMPTION_V2_MODEL", "").strip()
    if (
        api_key_hmac_commitment_v1(observed_key)
        != expected["api_key_hmac_sha256"]
        or _normalize_origin(observed_base) != expected["api_origin"]
        or observed_model != expected["model"]
    ):
        raise ProviderIdentityError("loaded provider environment drifted")
    return {
        **expected,
        "environment_aliases_scrubbed": True,
        "single_env_file_loaded": True,
        "dotenv_override": True,
    }


def _read_complete_canary_report_v1(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, raw = _read_json(path, label="provider canary report")
    expected_fields = {
        "canary_version",
        "model",
        "provider_chain",
        "provider_chain_hash",
        "root_hypothesis_id",
        "root_hypothesis_hash",
        "recursive_node_count",
        "recursive_depth",
        "accepted",
        "accepted_program",
        "nodes",
        "api_key_present",
        "secret_value_persisted",
        "raw_content_persisted",
    }
    nodes = payload.get("nodes")
    node_count = payload.get("recursive_node_count")
    if (
        set(payload) != expected_fields
        or payload.get("canary_version") != PROPOSAL_CANARY_VERSION
        or not isinstance(payload.get("model"), str)
        or not payload.get("model")
        or payload.get("provider_chain") != ["openai_compatible"]
        or payload.get("provider_chain_hash")
        != stable_hash(
            {
                "providers": ["openai_compatible"],
                "model": payload.get("model"),
            }
        )
        or not isinstance(payload.get("root_hypothesis_id"), str)
        or not payload.get("root_hypothesis_id")
        or not _is_sha256(payload.get("root_hypothesis_hash"))
        or isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count <= 0
        or isinstance(payload.get("recursive_depth"), bool)
        or not isinstance(payload.get("recursive_depth"), int)
        or payload.get("recursive_depth") < 0
        or not isinstance(nodes, list)
        or len(nodes) != node_count
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("hypothesis_id"), str)
            or not row.get("hypothesis_id")
            or not _is_sha256(row.get("hypothesis_hash"))
            or isinstance(row.get("depth"), bool)
            or not isinstance(row.get("depth"), int)
            or not isinstance(row.get("passed"), bool)
            or not isinstance(row.get("checks"), list)
            for row in nodes
        )
        or not isinstance(payload.get("accepted"), bool)
        or (
            payload.get("accepted") is True
            and not isinstance(payload.get("accepted_program"), Mapping)
        )
        or (
            payload.get("accepted") is False
            and payload.get("accepted_program") is not None
        )
        or payload.get("api_key_present") is not True
        or payload.get("secret_value_persisted") is not False
        or payload.get("raw_content_persisted") is not False
    ):
        raise ProviderIdentityError("provider canary response is incomplete")
    return payload, {
        "canary_report_file_sha256": hashlib.sha256(raw).hexdigest(),
        "canary_payload_hash": stable_hash(payload),
    }


def _read_event_ledger_v1(path: Path, *, model: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProviderIdentityError("provider event ledger is not regular")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProviderIdentityError("provider event ledger is unreadable") from exc
    if not raw or len(raw) > 32 * 1024 * 1024:
        raise ProviderIdentityError("provider event ledger is outside its bound")
    rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for line in text.splitlines():
        if not line:
            raise ProviderIdentityError("provider event ledger has a blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderIdentityError(
                "provider event ledger contains malformed JSON"
            ) from exc
        if not isinstance(row, dict) or not isinstance(
            row.get("payload"), Mapping
        ):
            raise ProviderIdentityError("provider event envelope drifted")
        reconstructed = Event(
            event=str(row.get("event") or ""),
            stage=str(row.get("stage") or ""),
            trace_id=str(row.get("trace_id") or ""),
            payload=dict(row["payload"]),
        ).to_dict()
        if row != reconstructed or row["event_id"] in event_ids:
            raise ProviderIdentityError("provider event integrity failed")
        event_ids.add(str(row["event_id"]))
        rows.append(row)
    selected = [
        row for row in rows if row["event"] == "model_provider_selected"
    ]
    if not selected or any(
        row["payload"].get("model") != model for row in selected
    ):
        raise ProviderIdentityError(
            "provider event ledger lacks the completed selected route"
        )
    return {
        "event_ledger_file_sha256": hashlib.sha256(raw).hexdigest(),
        "event_ledger_hash": stable_hash(rows),
        "event_count": len(rows),
    }


def build_provider_identity_sidecar_v1(
    *,
    provider_label: str,
    canary_report_path: str | Path,
    event_ledger_path: str | Path,
    env_file: str | Path,
) -> dict[str, Any]:
    if provider_label not in REGISTERED_PROVIDER_LABELS:
        raise ProviderIdentityError("provider label is not registered")
    identity = provider_environment_identity_from_file_v1(env_file)
    canary, canary_receipt = _read_complete_canary_report_v1(
        Path(canary_report_path)
    )
    event_receipt = _read_event_ledger_v1(
        Path(event_ledger_path), model=str(canary["model"])
    )
    if canary["model"] != identity["model"]:
        raise ProviderIdentityError("provider canary model differs from env")
    body = {
        "sidecar_version": PROVIDER_IDENTITY_SIDECAR_VERSION,
        "provider_label": provider_label,
        "provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "proposal_canary_version": PROPOSAL_CANARY_VERSION,
        **canary_receipt,
        **event_receipt,
        "model": identity["model"],
        "api_origin": identity["api_origin"],
        "api_key_commitment_version": API_KEY_COMMITMENT_VERSION,
        "api_key_commitment_challenge_sha256": hashlib.sha256(
            API_KEY_COMMITMENT_CHALLENGE
        ).hexdigest(),
        "api_key_hmac_sha256": identity["api_key_hmac_sha256"],
        "model_response_received": True,
        "semantic_acceptance_used_for_provider_selection": False,
        "environment_aliases_scrubbed_before_canary": True,
        "single_env_file_loaded": True,
        "env_file_content_persisted": False,
        "api_key_persisted": False,
        "secret_value_persisted": False,
        "raw_canary_content_persisted": False,
    }
    return {**body, "sidecar_hash": stable_hash(body)}


def validate_provider_identity_sidecar_v1(
    value: Mapping[str, Any],
    *,
    canary_report_path: str | Path,
    event_ledger_path: str | Path,
    env_file: str | Path | None = None,
    expected_provider_label: str | None = None,
) -> str:
    if set(value) != _SIDECAR_FIELDS:
        raise ProviderIdentityError("provider identity sidecar fields drifted")
    body = dict(value)
    sidecar_hash = body.pop("sidecar_hash", None)
    label = value.get("provider_label")
    if (
        sidecar_hash != stable_hash(body)
        or value.get("sidecar_version")
        != PROVIDER_IDENTITY_SIDECAR_VERSION
        or label not in REGISTERED_PROVIDER_LABELS
        or (
            expected_provider_label is not None
            and label != expected_provider_label
        )
        or value.get("provider_label_hash")
        != stable_hash({"provider_label": label})
        or value.get("proposal_canary_version")
        != PROPOSAL_CANARY_VERSION
        or value.get("api_key_commitment_version")
        != API_KEY_COMMITMENT_VERSION
        or value.get("api_key_commitment_challenge_sha256")
        != hashlib.sha256(API_KEY_COMMITMENT_CHALLENGE).hexdigest()
        or not _is_sha256(value.get("api_key_hmac_sha256"))
        or value.get("model_response_received") is not True
        or value.get("semantic_acceptance_used_for_provider_selection")
        is not False
        or value.get("environment_aliases_scrubbed_before_canary")
        is not True
        or value.get("single_env_file_loaded") is not True
        or value.get("env_file_content_persisted") is not False
        or value.get("api_key_persisted") is not False
        or value.get("secret_value_persisted") is not False
        or value.get("raw_canary_content_persisted") is not False
    ):
        raise ProviderIdentityError("provider identity sidecar drifted")
    canary, canary_receipt = _read_complete_canary_report_v1(
        Path(canary_report_path)
    )
    event_receipt = _read_event_ledger_v1(
        Path(event_ledger_path), model=str(canary["model"])
    )
    if (
        value.get("model") != canary.get("model")
        or value.get("canary_report_file_sha256")
        != canary_receipt["canary_report_file_sha256"]
        or value.get("canary_payload_hash")
        != canary_receipt["canary_payload_hash"]
        or any(
            value.get(key) != receipt_value
            for key, receipt_value in event_receipt.items()
        )
    ):
        raise ProviderIdentityError("provider sidecar evidence drifted")
    try:
        normalized_declared_origin = _normalize_origin(
            str(value.get("api_origin") or "")
        )
    except ProviderIdentityError:
        raise
    if normalized_declared_origin != value.get("api_origin"):
        raise ProviderIdentityError("provider sidecar origin is not canonical")
    if env_file is not None:
        identity = provider_environment_identity_from_file_v1(env_file)
        if (
            value.get("model") != identity["model"]
            or value.get("api_origin") != identity["api_origin"]
            or value.get("api_key_hmac_sha256")
            != identity["api_key_hmac_sha256"]
        ):
            raise ProviderIdentityError(
                "current provider env differs from pre-batch identity"
            )
    return str(sidecar_hash)


def write_provider_identity_sidecar_v1(
    output_path: str | Path,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(sidecar)
    declared = body.pop("sidecar_hash", None)
    if declared != stable_hash(body):
        raise ProviderIdentityError("provider sidecar self hash drifted")
    return atomic_write_hashed_json_v2(
        output_path,
        body,
        hash_field="sidecar_hash",
    )


def run_controlled_provider_canary_v1(
    *,
    project_root: str | Path,
    provider_label: str,
    env_file: str | Path,
    canary_report_path: str | Path,
    event_ledger_path: str | Path,
    sidecar_path: str | Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run one successful proposal canary in a scrubbed child process."""

    if provider_label not in REGISTERED_PROVIDER_LABELS:
        raise ProviderIdentityError("provider label is not registered")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ProviderIdentityError("provider canary timeout is invalid")
    project = Path(project_root).expanduser().resolve(strict=True)
    unresolved_env = Path(env_file).expanduser()
    if unresolved_env.is_symlink() or not unresolved_env.is_file():
        raise ProviderIdentityError("provider env file is not regular")
    env_source = unresolved_env.resolve(strict=True)
    # Validate before the child starts; this reads no ambient provider value.
    provider_environment_identity_from_file_v1(env_source)
    report = Path(canary_report_path).expanduser()
    events = Path(event_ledger_path).expanduser()
    sidecar_output = Path(sidecar_path).expanduser()
    outputs = (report, events, sidecar_output)
    if len({path.resolve() for path in outputs}) != len(outputs) or any(
        path.exists() or path.is_symlink() for path in outputs
    ):
        raise FileExistsError("provider canary output already exists")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise ProviderIdentityError(
                "provider canary output parent is not regular"
            )
    child_environment = dict(os.environ)
    scrub_provider_environment_v1(child_environment)
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "assumption_agent.proposal_canary",
                "--env-file",
                str(env_source),
                "--out",
                str(report),
                "--events",
                str(events),
            ],
            cwd=project,
            env=child_environment,
            capture_output=True,
            text=False,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderIdentityError("provider canary process failed") from exc
    if completed.returncode != 0:
        raise ProviderIdentityError(
            "provider canary did not produce a complete response"
        )
    sidecar = build_provider_identity_sidecar_v1(
        provider_label=provider_label,
        canary_report_path=report,
        event_ledger_path=events,
        env_file=env_source,
    )
    return write_provider_identity_sidecar_v1(sidecar_output, sidecar)


def _validate_selection_receipt_v1(
    path: Path,
    *,
    provider_label: str,
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    payload, raw = _read_json(path, label="provider selection receipt")
    expected_fields = {
        "selection_policy",
        "selected_provider_label",
        "selected_provider_label_hash",
        "probe_order",
        "plus_probe_receipt",
        "pro_model_response_receipt",
        "selected_model_response_receipt",
        "plus_semantic_acceptance_used_for_selection",
        "selection_completed_before_crossfit_task_calls",
        "crossfit_task_calls_before_selection",
        "crossfit_model_calls_before_selection",
        "selected_provider_fixed_for_complete_three_cell_batch",
        "mid_batch_provider_switch_authorized",
        "mid_batch_retry_authorized",
        "valid_failure_retry_authorized",
        "resampling_authorized",
        "secret_value_persisted",
        "raw_canary_content_persisted",
        "receipt_hash",
    }
    body = dict(payload)
    receipt_hash = body.pop("receipt_hash", None)
    selected = payload.get("selected_model_response_receipt")
    expected_selected = {
        "probe_kind": "complete_model_response_canary",
        "provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "canary_file_sha256": sidecar["canary_report_file_sha256"],
        "canary_payload_hash": sidecar["canary_payload_hash"],
        "model_response_received": True,
        "semantic_acceptance_used_for_provider_selection": False,
        "canary_semantic_accepted": (
            selected.get("canary_semantic_accepted")
            if isinstance(selected, Mapping)
            else None
        ),
        "raw_canary_content_persisted": False,
    }
    plus_probe = payload.get("plus_probe_receipt")
    plus_route = (
        provider_label == "plus"
        and payload.get("probe_order")
        == ["plus_complete_model_response"]
        and plus_probe == expected_selected
        and payload.get("pro_model_response_receipt") is None
    )
    pro_route = (
        provider_label == "pro"
        and payload.get("probe_order")
        == [
            "plus_pre_task_provider_unavailability",
            "pro_complete_model_response",
        ]
        and isinstance(plus_probe, Mapping)
        and set(plus_probe)
        == {
            "probe_kind",
            "receipt_file_sha256",
            "receipt_hash",
            "failure_summary",
            "model_response_received",
            "raw_failure_content_persisted",
        }
        and plus_probe.get("probe_kind")
        == "pre_task_provider_unavailability"
        and _is_sha256(plus_probe.get("receipt_file_sha256"))
        and _is_sha256(plus_probe.get("receipt_hash"))
        and isinstance(plus_probe.get("failure_summary"), Mapping)
        and plus_probe.get("model_response_received") is False
        and plus_probe.get("raw_failure_content_persisted") is False
        and payload.get("pro_model_response_receipt") == expected_selected
    )
    if (
        set(payload) != expected_fields
        or receipt_hash != stable_hash(body)
        or payload.get("selection_policy") != PROVIDER_SELECTION_POLICY
        or payload.get("selected_provider_label") != provider_label
        or payload.get("selected_provider_label_hash")
        != stable_hash({"provider_label": provider_label})
        or not isinstance(selected, Mapping)
        or not isinstance(selected.get("canary_semantic_accepted"), bool)
        or selected != expected_selected
        or not (plus_route or pro_route)
        or payload.get("plus_semantic_acceptance_used_for_selection")
        is not False
        or payload.get("selection_completed_before_crossfit_task_calls")
        is not True
        or payload.get("crossfit_task_calls_before_selection") != 0
        or payload.get("crossfit_model_calls_before_selection") != 0
        or payload.get(
            "selected_provider_fixed_for_complete_three_cell_batch"
        )
        is not True
        or payload.get("mid_batch_provider_switch_authorized") is not False
        or payload.get("mid_batch_retry_authorized") is not False
        or payload.get("valid_failure_retry_authorized") is not False
        or payload.get("resampling_authorized") is not False
        or payload.get("secret_value_persisted") is not False
        or payload.get("raw_canary_content_persisted") is not False
    ):
        raise ProviderIdentityError("provider selection is not authorized")
    return {
        "selection_receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "selection_receipt_hash": str(receipt_hash),
        "plus_transport_failure_before_pro_selection": (
            provider_label == "pro"
        ),
    }


def _project_file(
    project_root: Path,
    supplied: str | Path,
    *,
    label: str,
) -> tuple[Path, str]:
    unresolved = Path(supplied).expanduser()
    if not unresolved.is_absolute():
        unresolved = project_root / unresolved
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ProviderIdentityError(f"{label} is not a regular file")
    resolved = unresolved.resolve(strict=True)
    try:
        relative = resolved.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ProviderIdentityError(f"{label} escaped the project") from exc
    return resolved, relative


def build_execution_provider_binding_v1(
    *,
    project_root: str | Path,
    provider_label: str,
    identity_sidecar_path: str | Path,
    selected_canary_report_path: str | Path,
    selected_event_ledger_path: str | Path,
    selection_receipt_path: str | Path,
    env_file: str | Path | None = None,
) -> dict[str, Any]:
    if provider_label not in REGISTERED_PROVIDER_LABELS:
        raise ProviderIdentityError("provider label is not registered")
    project = Path(project_root).expanduser().resolve(strict=True)
    sidecar_path, sidecar_relative = _project_file(
        project, identity_sidecar_path, label="provider identity sidecar"
    )
    canary_path, canary_relative = _project_file(
        project, selected_canary_report_path, label="selected canary"
    )
    event_path, event_relative = _project_file(
        project, selected_event_ledger_path, label="selected event ledger"
    )
    selection_path, selection_relative = _project_file(
        project, selection_receipt_path, label="provider selection receipt"
    )
    sidecar, sidecar_raw = _read_json(
        sidecar_path, label="provider identity sidecar"
    )
    validate_provider_identity_sidecar_v1(
        sidecar,
        canary_report_path=canary_path,
        event_ledger_path=event_path,
        env_file=env_file,
        expected_provider_label=provider_label,
    )
    selection = _validate_selection_receipt_v1(
        selection_path,
        provider_label=provider_label,
        sidecar=sidecar,
    )
    body = {
        "binding_version": EXECUTION_PROVIDER_BINDING_VERSION,
        "provider_label": provider_label,
        "provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "fallback_policy": FALLBACK_POLICY,
        "identity_sidecar_relative_path": sidecar_relative,
        "identity_sidecar_file_sha256": hashlib.sha256(
            sidecar_raw
        ).hexdigest(),
        "identity_sidecar_hash": sidecar["sidecar_hash"],
        "selected_canary_relative_path": canary_relative,
        "selected_canary_file_sha256": sidecar[
            "canary_report_file_sha256"
        ],
        "selected_event_ledger_relative_path": event_relative,
        "selected_event_ledger_file_sha256": sidecar[
            "event_ledger_file_sha256"
        ],
        "selection_receipt_relative_path": selection_relative,
        "selection_receipt_file_sha256": selection[
            "selection_receipt_file_sha256"
        ],
        "selection_receipt_hash": selection["selection_receipt_hash"],
        "model": sidecar["model"],
        "api_origin": sidecar["api_origin"],
        "api_key_commitment_version": API_KEY_COMMITMENT_VERSION,
        "api_key_hmac_sha256": sidecar["api_key_hmac_sha256"],
        "plus_transport_failure_before_pro_selection": selection[
            "plus_transport_failure_before_pro_selection"
        ],
        "selected_provider_fixed_for_complete_batch": True,
        "mid_batch_provider_switch_authorized": False,
        "mid_batch_retry_authorized": False,
        "secret_value_persisted": False,
    }
    return {**body, "binding_hash": stable_hash(body)}


def validate_execution_provider_binding_v1(
    value: Mapping[str, Any],
    *,
    project_root: str | Path,
    env_file: str | Path | None = None,
) -> dict[str, Any]:
    if set(value) != _BINDING_FIELDS:
        raise ProviderIdentityError("execution provider binding fields drifted")
    project = Path(project_root).expanduser().resolve(strict=True)

    def bound_file(field: str) -> Path:
        relative = value.get(field)
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ProviderIdentityError("provider binding path is unsafe")
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise ProviderIdentityError("provider binding file is unavailable")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(project)
        except ValueError as exc:
            raise ProviderIdentityError(
                "provider binding path escaped project"
            ) from exc
        return resolved

    expected = build_execution_provider_binding_v1(
        project_root=project,
        provider_label=str(value.get("provider_label") or ""),
        identity_sidecar_path=bound_file(
            "identity_sidecar_relative_path"
        ),
        selected_canary_report_path=bound_file(
            "selected_canary_relative_path"
        ),
        selected_event_ledger_path=bound_file(
            "selected_event_ledger_relative_path"
        ),
        selection_receipt_path=bound_file(
            "selection_receipt_relative_path"
        ),
        env_file=env_file,
    )
    if dict(value) != expected:
        raise ProviderIdentityError("execution provider binding drifted")
    return {
        "provider_label": expected["provider_label"],
        "binding_hash": expected["binding_hash"],
        "identity_sidecar_hash": expected["identity_sidecar_hash"],
        "selection_receipt_hash": expected["selection_receipt_hash"],
        "model": expected["model"],
        "api_origin": expected["api_origin"],
        "api_key_commitment_version": expected[
            "api_key_commitment_version"
        ],
        "api_key_hmac_sha256": expected["api_key_hmac_sha256"],
        "plus_transport_failure_before_pro_selection": expected[
            "plus_transport_failure_before_pro_selection"
        ],
        "selected_provider_fixed_for_complete_batch": True,
        "mid_batch_provider_switch_authorized": False,
        "mid_batch_retry_authorized": False,
        "secret_value_persisted": False,
    }


__all__ = [
    "API_KEY_COMMITMENT_CHALLENGE",
    "API_KEY_COMMITMENT_VERSION",
    "EXECUTION_PROVIDER_BINDING_VERSION",
    "FALLBACK_POLICY",
    "PROVIDER_ENVIRONMENT_KEYS",
    "PROVIDER_IDENTITY_SIDECAR_VERSION",
    "ProviderIdentityError",
    "api_key_hmac_commitment_v1",
    "build_execution_provider_binding_v1",
    "build_provider_identity_sidecar_v1",
    "load_provider_environment_v1",
    "provider_environment_identity_from_file_v1",
    "run_controlled_provider_canary_v1",
    "scrub_provider_environment_v1",
    "validate_execution_provider_binding_v1",
    "validate_provider_identity_sidecar_v1",
    "write_provider_identity_sidecar_v1",
]
