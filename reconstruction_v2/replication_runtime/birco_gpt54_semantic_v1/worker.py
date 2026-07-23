"""One-request GPT-5.4 worker for the frozen BIRCO P1 semantic actions.

The caller supplies one canonical anonymous input envelope and an already
selected provider environment.  The worker performs exactly one HTTP request,
never retries, never switches provider, never persists the credential or raw
completion, and writes one exclusive terminal artifact.  Transport or output
failures become deterministic totalized actions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from . import contract


VERSION = "birco_p1_gpt54_semantic_worker_v1"
CANARY_SCHEMA = f"{VERSION}_constant_canary_input"
PROVIDER_ORIGIN = "https://ruoli.dev"
TRANSPORT_ID = "urllib_openai_compatible_chat_completions_one_request_v1"
KEY_COMMITMENT_VERSION = "birco_p1_provider_key_hmac_sha256_v1"
KEY_COMMITMENT_CHALLENGE = b"birco-p1/provider-key/formal-route/v1"
MAXIMUM_INPUT_BYTES = 4 * 1024 * 1024
MAXIMUM_WIRE_RESPONSE_BYTES = 4 * 1024 * 1024
MODEL_TIMEOUT_SECONDS = 600.0

_KEY_ALIASES = ("ASSUMPTION_V2_API_KEY", "RUOLI_GPT_KEY", "GPT5_API_KEY")
_BASE_ALIASES = ("ASSUMPTION_V2_API_BASE", "RUOLI_BASE_URL", "GPT5_BASE_URL")
_MODEL_ALIASES = ("ASSUMPTION_V2_MODEL",)
_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


class BircoP1GptWorkerError(RuntimeError):
    """The worker input, provider binding, or durable boundary drifted."""


@dataclass(frozen=True)
class Provider:
    api_base: str
    api_origin: str
    api_key: str = field(repr=False, compare=False)
    model: str = contract.MODEL_ID
    label: str = "selected"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.api_key, str)
            or not 1 <= len(self.api_key) <= 8_192
            or any(ord(character) < 33 or ord(character) > 126 for character in self.api_key)
        ):
            raise BircoP1GptWorkerError("provider key is not safe header text")
        if self.model != contract.MODEL_ID:
            raise BircoP1GptWorkerError("provider model differs from the frozen model")
        if _LABEL.fullmatch(self.label) is None:
            raise BircoP1GptWorkerError("provider label is unsafe")
        canonical_base, canonical_origin = _provider_base(self.api_base)
        if canonical_base != self.api_base or canonical_origin != self.api_origin:
            raise BircoP1GptWorkerError("provider route is not canonical")

    @property
    def key_hmac_sha256(self) -> str:
        return hmac.new(
            self.api_key.encode("utf-8"),
            KEY_COMMITMENT_CHALLENGE,
            hashlib.sha256,
        ).hexdigest()

    def safe_identity(self) -> dict[str, object]:
        return {
            "api_key_hmac_sha256": self.key_hmac_sha256,
            "api_origin": self.api_origin,
            "key_commitment_version": KEY_COMMITMENT_VERSION,
            "model": self.model,
            "provider_label": self.label,
            "secret_persisted": False,
        }


def _one_alias(names: Sequence[str], *, label: str, required: bool = True) -> str:
    present = [(name, os.environ[name].strip()) for name in names if os.environ.get(name, "").strip()]
    if not present:
        if required:
            raise BircoP1GptWorkerError(f"provider {label} is absent")
        return ""
    values = {value for _name, value in present}
    if len(values) != 1:
        raise BircoP1GptWorkerError(f"provider {label} aliases conflict")
    return present[0][1]


def _provider_base(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(value)
        port_value = parsed.port
    except ValueError as exc:
        raise BircoP1GptWorkerError("provider API base is malformed") from exc
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or path not in {"", "/v1"}
    ):
        raise BircoP1GptWorkerError("provider API base is outside the frozen route")
    port = f":{port_value}" if port_value is not None else ""
    origin = f"{parsed.scheme}://{parsed.hostname}{port}"
    if origin != PROVIDER_ORIGIN:
        raise BircoP1GptWorkerError("provider origin differs from the frozen origin")
    return origin + path, origin


def load_provider_from_environment() -> Provider:
    api_key = _one_alias(_KEY_ALIASES, label="key")
    api_base, origin = _provider_base(_one_alias(_BASE_ALIASES, label="base"))
    model = _one_alias(_MODEL_ALIASES, label="model", required=False) or contract.MODEL_ID
    if model != contract.MODEL_ID:
        raise BircoP1GptWorkerError("provider model differs from the frozen model")
    label = os.environ.get("BIRCO_P1_PROVIDER_LABEL", "selected").strip()
    if _LABEL.fullmatch(label) is None:
        raise BircoP1GptWorkerError("provider label is unsafe")
    return Provider(api_base=api_base, api_origin=origin, api_key=api_key, model=model, label=label)


def _read_canonical_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BircoP1GptWorkerError("worker input is unavailable")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAXIMUM_INPUT_BYTES:
        raise BircoP1GptWorkerError("worker input size or type drifted")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoP1GptWorkerError("worker input is invalid JSON") from exc
    if not isinstance(value, dict) or raw != contract.canonical_json_bytes(value):
        raise BircoP1GptWorkerError("worker input is not canonical JSON")
    return value


def _plan_from_payload(value: object) -> contract.Plan:
    if not isinstance(value, Mapping) or set(value) != {"edges", "facets", "generation_valid"}:
        raise BircoP1GptWorkerError("bound plan shape drifted")
    raw_facets = value.get("facets")
    raw_edges = value.get("edges")
    if (
        isinstance(raw_facets, (str, bytes))
        or not isinstance(raw_facets, Sequence)
        or isinstance(raw_edges, (str, bytes))
        or not isinstance(raw_edges, Sequence)
    ):
        raise BircoP1GptWorkerError("bound plan rows drifted")
    try:
        facets = tuple(
            contract.Facet(
                ordinal=row["ordinal"],
                facet_type=row["type"],
                text=row["text"],
                weight=row["weight"],
            )
            for row in raw_facets
            if isinstance(row, Mapping) and set(row) == {"ordinal", "text", "type", "weight"}
        )
        edges = tuple(
            contract.PlanEdge(source=row["source"], target=row["target"], edge_type=row["type"])
            for row in raw_edges
            if isinstance(row, Mapping) and set(row) == {"source", "target", "type"}
        )
        if len(facets) != len(raw_facets) or len(edges) != len(raw_edges):
            raise BircoP1GptWorkerError("bound plan row shape drifted")
        return contract.Plan(facets, edges, value.get("generation_valid"))
    except (KeyError, TypeError, contract.BircoP1GptContractError) as exc:
        raise BircoP1GptWorkerError("bound plan is invalid") from exc


def _candidates_from_payload(value: object) -> tuple[contract.CandidateProjection, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BircoP1GptWorkerError("candidate projections are absent")
    result: list[contract.CandidateProjection] = []
    try:
        for row in value:
            if not isinstance(row, Mapping) or set(row) != {"evidence_units", "ordinal", "text"}:
                raise BircoP1GptWorkerError("candidate projection shape drifted")
            raw_units = row.get("evidence_units")
            if isinstance(raw_units, (str, bytes)) or not isinstance(raw_units, Sequence):
                raise BircoP1GptWorkerError("candidate evidence rows drifted")
            units = tuple(
                contract.EvidenceUnit(
                    ordinal=unit["ordinal"],
                    byte_start=unit["byte_start"],
                    byte_end=unit["byte_end"],
                    text=unit["text"],
                )
                for unit in raw_units
                if isinstance(unit, Mapping)
                and set(unit) == {"byte_end", "byte_start", "ordinal", "text"}
            )
            if len(units) != len(raw_units):
                raise BircoP1GptWorkerError("candidate evidence shape drifted")
            projection = contract.CandidateProjection(row["ordinal"], units)
            if row.get("text") != projection.projection_text:
                raise BircoP1GptWorkerError("candidate common projection text drifted")
            result.append(projection)
    except (KeyError, TypeError, contract.BircoP1GptContractError) as exc:
        raise BircoP1GptWorkerError("candidate projection is invalid") from exc
    # Reuse the public constructor to enforce batch bounds and uniqueness.
    if not result:
        raise BircoP1GptWorkerError("candidate batch is empty")
    # Constructors above enforce each row; the formal batch slicing metadata
    # is validated by the mode-specific public input constructor below.
    if len(result) > contract.MAXIMUM_CANDIDATES_PER_BATCH:
        raise BircoP1GptWorkerError("candidate batch exceeds the frozen bound")
    return tuple(result)


def _validate_input(mode: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if mode == "canary":
        if value != {"fixed_probe": "complete_one_response", "schema": CANARY_SCHEMA}:
            raise BircoP1GptWorkerError("canary input drifted")
        return dict(value)
    if mode == "plan":
        if set(value) != {"objective", "query", "schema", "work_id"} or value.get("schema") != contract.PLAN_INPUT_SCHEMA:
            raise BircoP1GptWorkerError("planner input shape drifted")
        return contract.planner_input(
            work_id=value.get("work_id"),
            objective=value.get("objective"),
            query=value.get("query"),
        )
    if mode == "matrix":
        if set(value) != {
            "batch_count",
            "batch_common_projection_sha256",
            "batch_ordinal",
            "candidates",
            "objective",
            "plan",
            "pool_candidate_count",
            "pool_common_projection_sha256",
            "query",
            "schema",
            "work_id",
        } or value.get("schema") != contract.MATRIX_INPUT_SCHEMA:
            raise BircoP1GptWorkerError("matrix input shape drifted")
        plan = _plan_from_payload(value.get("plan"))
        candidates = _candidates_from_payload(value.get("candidates"))
        canonical = contract.matrix_input(
            work_id=value.get("work_id"),
            objective=value.get("objective"),
            query=value.get("query"),
            plan=plan,
            candidates=candidates,
            batch_ordinal=value.get("batch_ordinal"),
            batch_count=value.get("batch_count"),
            pool_candidate_count=value.get("pool_candidate_count"),
            pool_common_projection_sha256=value.get(
                "pool_common_projection_sha256"
            ),
        )
        if canonical["batch_common_projection_sha256"] != value.get(
            "batch_common_projection_sha256"
        ):
            raise BircoP1GptWorkerError(
                "matrix batch projection commitment drifted"
            )
        return canonical
    if mode == "raw":
        if set(value) != {
            "batch_count",
            "batch_common_projection_sha256",
            "batch_ordinal",
            "candidates",
            "objective",
            "pool_candidate_count",
            "pool_common_projection_sha256",
            "query",
            "schema",
            "work_id",
        } or value.get("schema") != contract.RAW_INPUT_SCHEMA:
            raise BircoP1GptWorkerError("RAW input shape drifted")
        candidates = _candidates_from_payload(value.get("candidates"))
        canonical = contract.raw_input(
            work_id=value.get("work_id"),
            objective=value.get("objective"),
            query=value.get("query"),
            candidates=candidates,
            batch_ordinal=value.get("batch_ordinal"),
            batch_count=value.get("batch_count"),
            pool_candidate_count=value.get("pool_candidate_count"),
            pool_common_projection_sha256=value.get(
                "pool_common_projection_sha256"
            ),
        )
        if canonical["batch_common_projection_sha256"] != value.get(
            "batch_common_projection_sha256"
        ):
            raise BircoP1GptWorkerError("RAW batch projection commitment drifted")
        return canonical
    raise BircoP1GptWorkerError("worker mode is unknown")


_CANARY_SYSTEM = (
    "This is a fixed transport-only availability canary with no study content. "
    "Return one non-empty JSON object. Do not use tools or external context."
)
_PLAN_SYSTEM = (
    "You are a label-free typed retrieval planner. Using only objective and query, "
    "return exactly one JSON object with fields facets and edges. Emit 2-12 facets; "
    "each facet is {ordinal,type,text,weight}, with contiguous ordinals, type in "
    "REQUIRED/EXCLUDED/PREFERRED/ELIGIBILITY/TEMPORAL/RELATIONAL and integer weight "
    "1-4. Each edge is {source,target,type}, with type REQUIRES/REFINES/CONTRASTS_WITH; "
    "REQUIRES and REFINES must form an acyclic graph. No explanation or markdown."
)
_MATRIX_SYSTEM = (
    "You are a label-free facet-to-evidence scorer. Use only the supplied objective, "
    "query, typed plan, and anonymous candidate evidence. Return exactly "
    "{\"candidates\":[{\"ordinal\":N,\"rows\":[[S,C,E],...]}]}. Return every "
    "candidate once and one row per facet in facet order. S and C are integer support "
    "and contradiction strengths 0-4. E is one evidence-unit ordinal or null. Judge "
    "semantic entailment and contradiction, not term overlap. No explanation or markdown."
)
_RAW_SYSTEM = (
    "You are a direct relevance scorer without planning, facets, dependencies, or "
    "assignment. Use only objective, query, and anonymous candidate evidence. Return "
    "exactly {\"scores\":[{\"ordinal\":N,\"score\":S}]} with every candidate once "
    "and integer relevance S from 0 to 100. No explanation or markdown."
)


def _model_payload(mode: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove controller identity/accounting fields from the model-visible view."""

    if mode == "canary":
        return dict(payload)
    if mode == "plan":
        return {
            "objective": payload["objective"],
            "query": payload["query"],
            "schema": contract.PLAN_INPUT_SCHEMA,
        }
    common = {
        "candidates": payload["candidates"],
        "objective": payload["objective"],
        "query": payload["query"],
        "schema": payload["schema"],
    }
    if mode == "matrix":
        common["plan"] = payload["plan"]
    return common


def _request_body(mode: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    system = {
        "canary": _CANARY_SYSTEM,
        "plan": _PLAN_SYSTEM,
        "matrix": _MATRIX_SYSTEM,
        "raw": _RAW_SYSTEM,
    }[mode]
    visible = _model_payload(mode, payload)
    body = {
        "max_tokens": 128 if mode == "canary" else contract.MAXIMUM_OUTPUT_TOKENS,
        "messages": [
            {"content": system, "role": "system"},
            {
                "content": json.dumps(visible, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")),
                "role": "user",
            },
        ],
        "model": contract.MODEL_ID,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    raw = contract.canonical_json_bytes(body, newline=False)
    if len(raw) > MAXIMUM_INPUT_BYTES:
        raise BircoP1GptWorkerError("model request exceeds the frozen byte bound")
    return body


def _message_content(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise BircoP1GptWorkerError("provider response root is malformed")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BircoP1GptWorkerError("provider response has no choice")
    first = choices[0]
    if not isinstance(first, Mapping) or not isinstance(first.get("message"), Mapping):
        raise BircoP1GptWorkerError("provider response message is malformed")
    content = first["message"].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        joined = "".join(
            str(part.get("text") or "") for part in content if isinstance(part, Mapping)
        )
        if joined:
            return joined
    raise BircoP1GptWorkerError("provider response has no textual content")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Make one logical attempt exactly one HTTP request and keep auth on-origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _open_no_redirect(request: urllib.request.Request, *, timeout: float):
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


def _one_request(provider: Provider, body: Mapping[str, Any]) -> str:
    base = provider.api_base.rstrip("/")
    endpoint = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=contract.canonical_json_bytes(body, newline=False),
        headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with _open_no_redirect(request, timeout=MODEL_TIMEOUT_SECONDS) as response:
        raw = response.read(MAXIMUM_WIRE_RESPONSE_BYTES + 1)
    if not raw or len(raw) > MAXIMUM_WIRE_RESPONSE_BYTES:
        raise BircoP1GptWorkerError("provider response is empty or oversized")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoP1GptWorkerError("provider wire JSON is malformed") from exc
    return _message_content(value)


def _action(mode: str, payload: Mapping[str, Any], content: str) -> tuple[dict[str, Any], bool]:
    if mode == "canary":
        valid = bool(content.strip())
        return {"nonempty_response": valid}, valid
    if mode == "plan":
        plan = contract.parse_plan_completion(content, query=str(payload["query"]))
        return {"plan": plan.payload()}, plan.generation_valid
    candidates = _candidates_from_payload(payload["candidates"])
    if mode == "matrix":
        plan = _plan_from_payload(payload["plan"])
        matrix, valid = contract.parse_matrix_completion(content, plan=plan, candidates=candidates)
        return {"matrix": contract.matrix_payload(matrix)}, valid
    scores, valid = contract.parse_raw_completion(content, candidates=candidates)
    return {"scores": contract.raw_payload(scores)}, valid


def _transport_totalizer(mode: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if mode == "canary":
        return {"nonempty_response": False}
    if mode == "plan":
        return {"plan": contract.deterministic_plan_totalizer(str(payload["query"])).payload()}
    candidates = _candidates_from_payload(payload["candidates"])
    if mode == "matrix":
        plan = _plan_from_payload(payload["plan"])
        return {"matrix": contract.matrix_payload(contract.totalized_matrix(plan=plan, candidates=candidates))}
    return {"scores": contract.raw_payload(contract.totalized_raw(candidates))}


def execute_one(*, mode: str, payload: Mapping[str, Any], provider: Provider) -> dict[str, Any]:
    validated = _validate_input(mode, payload)
    body = _request_body(mode, validated)
    request_sha256 = hashlib.sha256(contract.canonical_json_bytes(body, newline=False)).hexdigest()
    response_sha256: str | None = None
    transport_succeeded = False
    generation_valid = False
    terminal_category = "transport_unavailable"
    try:
        content = _one_request(provider, body)
        transport_succeeded = True
        response_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        action, generation_valid = _action(mode, validated, content)
        terminal_category = "success" if generation_valid else "output_totalized"
    except (
        OSError,
        TimeoutError,
        ValueError,
        urllib.error.URLError,
        urllib.error.HTTPError,
    ):
        action = _transport_totalizer(mode, validated)
    except BircoP1GptWorkerError:
        # The request was consumed and a response may have been malformed.  It
        # remains a terminal, totalized action and is never resubmitted.
        transport_succeeded = True
        action = _transport_totalizer(mode, validated)
        terminal_category = "provider_protocol_totalized"

    metadata: dict[str, object] = {"work_id": validated.get("work_id")}
    if mode in {"matrix", "raw"}:
        metadata.update(
            {
                "batch_count": validated["batch_count"],
                "batch_ordinal": validated["batch_ordinal"],
                "batch_common_projection_sha256": validated[
                    "batch_common_projection_sha256"
                ],
                "pool_candidate_count": validated["pool_candidate_count"],
                "pool_common_projection_sha256": validated[
                    "pool_common_projection_sha256"
                ],
            }
        )
    body_without_hash = {
        "action": action,
        "attempt_count": 1,
        "generation_valid": generation_valid,
        "input_sha256": contract.semantic_hash(validated),
        "mode": mode,
        "model_request_sha256": request_sha256,
        "provider": provider.safe_identity(),
        "raw_completion_persisted": False,
        "response_sha256": response_sha256,
        "retry_replay_resample_or_provider_switch_count": 0,
        "schema": contract.TERMINAL_OUTPUT_SCHEMA,
        "terminal_category": terminal_category,
        "transport": TRANSPORT_ID,
        "transport_succeeded": transport_succeeded,
        **metadata,
    }
    return {**body_without_hash, "self_sha256": contract.semantic_hash(body_without_hash)}


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = contract.canonical_json_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    parent = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _attempt_claim(
    *, mode: str, payload: Mapping[str, Any], provider: Provider
) -> dict[str, Any]:
    body = {
        "input_sha256": contract.semantic_hash(payload),
        "mode": mode,
        "provider": provider.safe_identity(),
        "schema": f"{VERSION}_durable_pre_http_attempt_claim_v1",
        "status": "consumed_before_the_only_authorized_HTTP_request",
        "work_id": payload.get("work_id"),
    }
    return {**body, "self_sha256": contract.semantic_hash(body)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "plan", "matrix", "raw"), required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    payload = _read_canonical_object(arguments.input)
    provider = load_provider_from_environment()
    validated = _validate_input(arguments.mode, payload)
    claim_path = arguments.output.with_name(arguments.output.name + ".attempt.json")
    _write_exclusive(
        claim_path,
        _attempt_claim(mode=arguments.mode, payload=validated, provider=provider),
    )
    terminal = execute_one(
        mode=arguments.mode, payload=validated, provider=provider
    )
    _write_exclusive(arguments.output, terminal)
    print(
        json.dumps(
            {
                "generation_valid": terminal["generation_valid"],
                "mode": arguments.mode,
                "terminal_category": terminal["terminal_category"],
                "transport_succeeded": terminal["transport_succeeded"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
