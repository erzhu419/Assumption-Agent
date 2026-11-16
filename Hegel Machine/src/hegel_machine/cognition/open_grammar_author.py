"""Open-grammar LLM hypothesis authoring gated by PremiseDependencyGraph (v10 Phase 2B).

Implements a strictly controlled single-factor comparison between:
1. ``propose_flat_unconstrained_programs`` (``llm_without_premise_gate``):
   The LLM sees raw anonymous ``m1_*`` sensor observations, transitions, and flat
   scalar MSE, proposing ``Program`` ASTs directly over raw features without
   ``PremiseDependencyGraph`` premise localization or ``world_structure_commitments``.
2. ``propose_premise_gated_commitment_revisions`` (``llm_with_v10_premise_gate``):
   ``PremiseDependencyGraph`` first verifies upstream support premises and
   isolates the specific violated node (``P_exchange_form`` on paid transitions
   or ``P_temporal_readout`` on role-resolved response residuals). The LLM
   authors open-grammar parametric AST patches (custom ``edge_predicate`` over
   ``EDGE_ARGUMENTS`` or role-uniform ``all_object_sum`` readout basis terms),
   which must pass ``screen_custom_edge_predicate_on_transitions``, ``_Fitter``,
   and ``check_equivalence_reorganization`` (``Rej(T) >= Rej(S)``) before
   consolidation and zero-LLM-call cross-scale transfer.

Accounting is strictly separated:
- Every symbolic evaluation and fit charges ``SearchBudget`` (``compute_cost``).
- Every LLM invocation is recorded in ``llm_ledger`` (``model_calls``,
  ``input_tokens``, ``output_tokens``, ``total_tokens``, ``wall_time_seconds``),
  never hidden inside free host logic and never leaking API keys.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from time import monotonic
from typing import Callable
import urllib.error
import urllib.request

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import initial_prediction, loss
from .language import EvaluationContext, Program
from .models import Candidate, Observation, SearchBudget
from .premise_dependency_audit import (
    _audit_focus_membership_from_trajectories,
    _enumerate_position_bijections,
    apply_verified_transition_premises,
    audit_transition_premises,
    extract_necessary_active_transfer_geometry,
    screen_custom_edge_predicate_on_transitions,
    screen_custom_latent_transition_drive,
)
from .relation_language import EDGE_ARGUMENTS
from .research_programs import _Fitter
from .structure_lab import ObjectSchema
from .world_structure_commitments import (
    _as_schema,
    _extract_latent_poles_from_role_terms,
    _rebind_latent_pole_in_role_terms,
    expand_role_terms_for_custom_latent_transition,
    instantiate_commitment_on_schema,
    instantiate_commitment_terms,
)


ALLOWED_EDGE_PREDICATE_OPS = frozenset(
    {
        "const",
        "arg",
        "abs",
        "not",
        "add",
        "sub",
        "mul",
        "min",
        "max",
        "ge",
        "lt",
        "and",
        "or",
        "if",
    }
)

ALLOWED_ROLE_SUMMAND_OPS = frozenset(
    {
        "const",
        "feature",
        "accumulate",
        "abs",
        "not",
        "add",
        "sub",
        "mul",
        "min",
        "max",
        "ge",
        "lt",
        "and",
        "or",
        "if",
    }
)

ALLOWED_OBJECT_DRIVE_OPS = frozenset(
    {
        "const",
        "feature",
        "abs",
        "not",
        "add",
        "sub",
        "mul",
        "min",
        "max",
        "ge",
        "lt",
        "and",
        "or",
        "if",
    }
)

ALLOWED_OBJECT_ROLE_FEATURES = frozenset(
    {
        "object_role:q",
        "object_role:u",
        "object_role:x",
        "object_role:f",
    }
)


def load_dotenv_vars(env_path: str | Path | None = None) -> dict[str, str]:
    """Read key=value pairs from ``.env`` in memory without mutating ``os.environ`` or logging."""
    candidates = []
    if env_path is not None:
        candidates.append(Path(env_path))
    # Workspace parent ``Asumption Agent/.env`` or repo ``.env``
    here = Path(__file__).resolve()
    candidates.extend([here.parents[4] / ".env", here.parents[3] / ".env"])
    parsed: dict[str, str] = {}
    for path in candidates:
        if path.is_file():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key and key not in parsed:
                    parsed[key] = val
            break
    return parsed


def empty_llm_ledger(model: str | None = None, backend: str = "none") -> dict:
    return {
        "backend": backend,
        "model": model,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "wall_time_seconds": 0.0,
        "call_records": [],
    }


def _record_usage(ledger: dict, usage: dict) -> None:
    ledger["model_calls"] += int(usage.get("calls", 1))
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        if isinstance(usage.get(k), int):
            ledger[k] += int(usage[k])
    ledger["wall_time_seconds"] += float(usage.get("wall_time_seconds") or 0.0)
    sanitized = {
        k: v
        for k, v in usage.items()
        if k
        in {
            "backend",
            "model",
            "returned_model",
            "calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "wall_time_seconds",
            "purpose",
            "status",
            "error",
        }
    }
    ledger["call_records"].append(sanitized)


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    fallback_dict: dict | None = None
    idx = 0
    while True:
        brace = stripped.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end_pos = decoder.raw_decode(stripped, brace)
            if isinstance(obj, dict):
                if "candidates" in obj or "hypotheses" in obj:
                    return obj
                if fallback_dict is None:
                    fallback_dict = obj
            idx = max(brace + 1, end_pos)
        except json.JSONDecodeError:
            idx = brace + 1
    if fallback_dict is not None:
        return fallback_dict
    raise ValueError("LLM response did not contain a valid JSON object")


class ResponsesJSONTransport:
    """Streaming ``/v1/responses`` JSON client with zero secret persistence."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        base_url: str | None = None,
        api_key_env: str = "RUOLI_GPT_KEY",
        env_path: str | Path | None = None,
        timeout_seconds: int = 120,
        max_output_tokens: int = 8192,
        reasoning_effort: str | None = None,
    ):
        dotenv = load_dotenv_vars(env_path)
        resolved_url = (
            base_url
            or os.environ.get("RUOLI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or dotenv.get("RUOLI_BASE_URL")
            or dotenv.get("OPENAI_BASE_URL")
        )
        if not resolved_url:
            raise ValueError("Responses base_url not configured in environment or .env")
        self.endpoint = resolved_url.rstrip("/")
        if not self.endpoint.endswith("/responses"):
            if not self.endpoint.endswith("/v1"):
                self.endpoint += "/v1"
            self.endpoint += "/responses"
        self._api_key = (
            os.environ.get(api_key_env)
            or os.environ.get("OPENAI_API_KEY")
            or dotenv.get(api_key_env)
            or dotenv.get("OPENAI_API_KEY")
            or ""
        )
        if not self._api_key:
            raise ValueError(f"API key {api_key_env} not found in environment or .env")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def __call__(self, instructions: str, payload: dict, *, purpose: str = "propose") -> tuple[dict | None, dict]:
        body: dict = {
            "model": self.model,
            "instructions": instructions,
            "stream": True,
            "input": json.dumps(payload, ensure_ascii=True),
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        usage = {
            "backend": "responses_http",
            "model": self.model,
            "returned_model": None,
            "calls": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "wall_time_seconds": 0.0,
            "purpose": purpose,
            "status": "pending",
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": "curl/8.0",
            },
            method="POST",
        )
        started = monotonic()
        completed_response = None
        streamed_chunks: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                for raw_line in resp:
                    if not raw_line.startswith(b"data:"):
                        continue
                    data = raw_line[5:].strip()
                    if not data or data == b"[DONE]":
                        if data == b"[DONE]":
                            break
                        continue
                    event = json.loads(data)
                    ev_type = event.get("type", "")
                    if ev_type == "response.output_text.delta":
                        streamed_chunks.append(event.get("delta", ""))
                    elif ev_type in {"response.completed", "response.incomplete", "response.failed"}:
                        completed_response = event.get("response")
                        break
                    elif ev_type == "error":
                        raise ValueError(str(event.get("message", "stream error")))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-1000:]
            usage["wall_time_seconds"] = monotonic() - started
            usage["status"] = "http_error"
            usage["error"] = f"HTTP {exc.code}: {detail}"
            return None, usage
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            usage["wall_time_seconds"] = monotonic() - started
            usage["status"] = "transport_error"
            usage["error"] = f"{type(exc).__name__}: {exc}"
            return None, usage

        usage["wall_time_seconds"] = monotonic() - started
        if completed_response is not None:
            usage["returned_model"] = completed_response.get("model")
            rep = completed_response.get("usage") or {}
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                if isinstance(rep.get(k), int):
                    usage[k] = int(rep[k])
            texts = [
                part.get("text", "")
                for item in completed_response.get("output", ())
                if item.get("type") == "message"
                for part in item.get("content", ())
                if part.get("type") in {"output_text", "text"}
            ]
            raw_text = "\n".join(t for t in texts if t).strip() or "".join(streamed_chunks).strip()
        else:
            raw_text = "".join(streamed_chunks).strip()

        if not raw_text:
            usage["status"] = "empty_output"
            return None, usage
        try:
            parsed = _extract_json_object(raw_text)
            usage["status"] = "ok"
            return parsed, usage
        except ValueError as exc:
            usage["status"] = "json_decode_error"
            usage["error"] = str(exc)
            return None, usage


def _normalize_program_dict(node: dict) -> dict:
    """Canonicalize common AST aliases equally across all arms."""
    if not isinstance(node, dict):
        return node
    op = node.get("op")
    raw_children = node.get("children") or []
    children = [_normalize_program_dict(c) for c in raw_children if isinstance(c, dict)]
    if op == "gt" and len(children) == 2:
        return {
            "op": "lt",
            "value": None,
            "name": None,
            "lag": 0,
            "children": [children[1], children[0]],
        }
    if op == "le" and len(children) == 2:
        return {
            "op": "ge",
            "value": None,
            "name": None,
            "lag": 0,
            "children": [children[1], children[0]],
        }
    if op == "neg" and len(children) == 1:
        return {
            "op": "sub",
            "value": None,
            "name": None,
            "lag": 0,
            "children": [
                {"op": "const", "value": 0.0, "name": None, "lag": 0, "children": []},
                children[0],
            ],
        }
    if op in {"sq", "square"} and len(children) == 1:
        return {
            "op": "mul",
            "value": None,
            "name": None,
            "lag": 0,
            "children": [deepcopy(children[0]), deepcopy(children[0])],
        }
    if op == "eq" and len(children) == 2:
        return {
            "op": "and",
            "value": None,
            "name": None,
            "lag": 0,
            "children": [
                {"op": "ge", "value": None, "name": None, "lag": 0, "children": [children[0], children[1]]},
                {"op": "ge", "value": None, "name": None, "lag": 0, "children": [deepcopy(children[1]), deepcopy(children[0])]},
            ],
        }
    if op in {"add", "mul", "and", "or", "min", "max"} and len(children) > 2:
        acc = children[0]
        for nxt in children[1:]:
            acc = {
                "op": op,
                "value": None,
                "name": None,
                "lag": 0,
                "children": [acc, nxt],
            }
        return acc
    out = {
        "op": op,
        "value": float(node["value"]) if node.get("value") is not None else None,
        "name": node.get("name"),
        "lag": int(node.get("lag") or 0),
        "children": children,
    }
    return out


def validate_edge_predicate_ast(payload: dict) -> Program | None:
    """Validate that ``payload`` is a safe lexical edge predicate over ``EDGE_ARGUMENTS``."""
    try:
        prog = Program.from_dict(_normalize_program_dict(payload))
    except (ValueError, TypeError, KeyError):
        return None

    def _ok(node: Program) -> bool:
        if node.op not in ALLOWED_EDGE_PREDICATE_OPS:
            return False
        if node.op == "arg" and node.name not in EDGE_ARGUMENTS:
            return False
        return all(_ok(child) for child in node.children)

    if not _ok(prog):
        return None
    # Smoke-test evaluation on one synthetic edge binding
    probe_bound = prog._expand(
        {
            "edge_source_position": Program("const", value=0.0),
            "edge_source_quantity": Program("const", value=0.5),
            "edge_target_position": Program("const", value=0.4),
            "edge_target_quantity": Program("const", value=0.1),
        }
    )
    try:
        val = EvaluationContext(Observation({}, 0.0)).evaluate(probe_bound)
    except Exception:
        return None
    if not isinstance(val, (int, float)):
        return None
    return prog


def validate_object_drive_ast(payload: dict) -> Program | None:
    """Validate an instantaneous per-object drive AST over ``object_role:{q, u, x}``."""
    try:
        prog = Program.from_dict(_normalize_program_dict(payload))
    except (ValueError, TypeError, KeyError):
        return None
    allowed_drive_feats = {"object_role:q", "object_role:u", "object_role:x"}

    def _ok(node: Program) -> bool:
        if node.op not in ALLOWED_OBJECT_DRIVE_OPS:
            return False
        if node.op == "feature" and node.name not in allowed_drive_feats:
            return False
        return all(_ok(child) for child in node.children)

    if not _ok(prog) or not prog.required_features():
        return None
    try:
        val = prog.predict(
            Observation({"object_role:q": 0.4, "object_role:u": -0.2, "object_role:x": 0.3}, 0.0)
        )
    except Exception:
        return None
    if not isinstance(val, (int, float)):
        return None
    return prog


def validate_role_term_spec(role_term: dict, probe_schema: ObjectSchema) -> dict | None:
    """Validate an open-grammar role-uniform basis term (``all_object_sum`` + temporal/accumulate ops)."""
    if not isinstance(role_term, dict):
        return None

    def _check_summand(node: Program) -> bool:
        if node.op not in ALLOWED_ROLE_SUMMAND_OPS:
            return False
        if node.op == "feature" and node.name not in ALLOWED_OBJECT_ROLE_FEATURES:
            return False
        if node.op == "accumulate":
            if (
                node.name is not None
                or node.value is None
                or not (-0.995 <= float(node.value) <= 0.995)
                or len(node.children) != 2
                or node.children[1].op != "const"
            ):
                return False
            return _check_summand(node.children[0])
        return all(_check_summand(child) for child in node.children)

    def _normalize(node: dict) -> dict | None:
        kind = node.get("kind")
        if kind == "all_object_sum":
            summand_raw = node.get("summand")
            if not isinstance(summand_raw, dict):
                return None
            try:
                summand_prog = Program.from_dict(_normalize_program_dict(summand_raw))
            except (ValueError, TypeError, KeyError):
                return None
            if not _check_summand(summand_prog):
                return None
            roles = sorted(
                name.removeprefix("object_role:") for name in summand_prog.required_features()
            )
            if not roles:
                return None
            return {
                "kind": "all_object_sum",
                "summand": summand_prog.to_dict(),
                "roles": roles,
            }
        if kind == "program_op":
            op = node.get("op")
            lag = int(node.get("lag", 0))
            children_raw = node.get("children") or []
            if op in {"window_sum", "event_delta"}:
                if lag < 1 or lag > 6 or len(children_raw) != 1:
                    return None
                child = _normalize(children_raw[0])
                if child is None:
                    return None
                return {
                    "kind": "program_op",
                    "op": op,
                    "name": None,
                    "value": None,
                    "lag": lag,
                    "children": [child],
                }
            if op in {"add", "sub", "mul"} and len(children_raw) == 2:
                left = _normalize(children_raw[0])
                right = _normalize(children_raw[1])
                if left is None or right is None:
                    return None
                return {
                    "kind": "program_op",
                    "op": op,
                    "name": None,
                    "value": None,
                    "lag": 0,
                    "children": [left, right],
                }
        return None

    normalized = _normalize(role_term)
    if normalized is None:
        return None
    dummy_commitment = {"role_terms": [normalized], "radius_parameters": []}
    instantiated = instantiate_commitment_terms(dummy_commitment, probe_schema)
    if instantiated is None:
        return None
    return normalized


def validate_flat_program_ast(payload: dict, allowed_features: set[str]) -> Program | None:
    """Validate an unconstrained flat ``Program`` AST over ``allowed_features``."""
    try:
        prog = Program.from_dict(_normalize_program_dict(payload))
    except (ValueError, TypeError, KeyError):
        return None
    for feat, _ in prog.observed_variables():
        if feat != "__target__" and feat not in allowed_features:
            return None
    if prog.minimum_history_depth > 0:
        return None
    return prog


_FLAT_AUTHOR_INSTRUCTIONS = """You are an executable scientific hypothesis proposer for Hegel Machine.
Given anonymous sensor observations (features m1_*, scalar target, short history) and raw before->after sensor transitions, propose up to 3 candidate basis-term sets in the Program JSON AST grammar to linearly predict `target`.

Program JSON node format:
{"op": <op>, "value": <float|null>, "name": <str|null>, "lag": <int>, "children": [...]}
Valid ops:
- Leaves: {"op": "const", "value": 0.25, "name": null, "lag": 0, "children": []}, {"op": "feature", "value": null, "name": "m1_00", "lag": 0, "children": []}
- Unary: "abs", "not" (lag=0, 1 child); temporal: "window_sum" (lag>=1, 1 child), "event_delta" (lag>=1, 1 child)
- Recurrent latent accumulator: {"op": "accumulate", "value": 0.65, "name": null, "lag": 0, "children": [<drive_AST>, {"op": "const", "value": 0.0, "name": null, "lag": 0, "children": []}]}
- Binary (lag=0, 2 children): "add", "sub", "mul", "min", "max", "ge" (left>=right), "lt" (left<right), "and", "or"
- Ternary: "if" (3 children: cond, then, else)
- Variadic: "object_sum" (children=[...])

Return ONLY valid JSON matching:
{"hypotheses": [{"rationale": "...", "basis_terms": [<Program_AST>, ...]}]}"""


_LATENT_TRANSITION_DRIVE_INSTRUCTIONS = """You are the open-grammar POMDP hypothesis author inside Hegel Machine v10, gated by PremiseDependencyGraph.
Premise P_role_partition (sensor roles q, u, x, f) and probe-reset spatial exchange (t == 0) are EMPIRICALLY VERIFIED!
However, on history steps (t >= 1), P_markov_state and P_latent_order (all linear reservoirs in q and u) are REFUTED:
each object i carries an unobserved internal reservoir state z_{i,t} = alpha * z_{i,t-1} + drive(q_{i,t}, u_{i,t}, x_{i,t}) (with z_{i,-1} = 0 at probe reset t=0), and `after[q_i]` receives `eta * z_{i,t-1}` at step t!
Notice in `deflated_trajectory_samples`:
- At step 0: `deflated_latent_injection_per_object` is `[0, 0, ...]` because `z_{i,-1} = 0`.
- At step 1: `deflated_latent_injection_per_object[i]` equals `eta * drive(q_{i,0}, u_{i,0}, x_{i,0})`!
- At step 2: `deflated_latent_injection_per_object[i]` equals `eta * (alpha * drive_0 + drive_1)`!
Inspect step 0 `(q, u, x)` vs step 1 `deflated_latent_injection_per_object` across objects to deduce the exact nonlinear algebraic `drive` AST!

Program JSON node format for `latent_drive_ast` (features restricted to `"object_role:q"`, `"object_role:u"`, `"object_role:x"`):
{"op": <op>, "value": <float|null>, "name": <str|null>, "lag": 0, "children": [...]}
- Feature leaves:
  * {"op": "feature", "value": null, "name": "object_role:q", "lag": 0, "children": []}
  * {"op": "feature", "value": null, "name": "object_role:u", "lag": 0, "children": []}
  * {"op": "feature", "value": null, "name": "object_role:x", "lag": 0, "children": []}
- Constant leaf: {"op": "const", "value": 0.1, "name": null, "lag": 0, "children": []}
- Ops: unary `"abs"`, `"not"`; binary `"add"`, `"sub"`, `"mul"`, `"min"`, `"max"`, `"lt"`, `"ge"`, `"and"`, `"or"`; ternary `"if"`.
Neutral syntax example (`abs(u_i)`):
{"op": "abs", "value": null, "name": null, "lag": 0, "children": [{"op": "feature", "value": null, "name": "object_role:u", "lag": 0, "children": []}]}

Return ONLY valid JSON matching (propose up to 4 candidates):
{"candidates": [{"rationale": "...", "latent_drive_ast": <Program_AST>}]}"""


_EDGE_PREDICATE_INSTRUCTIONS = """You are the open-grammar hypothesis author inside Hegel Machine v10, gated by PremiseDependencyGraph.
Upstream premises P_role_partition (sensor roles q, u, x, f) and P_conservation (sum_i(q'_i - q_i - u_i) == 0) are EMPIRICALLY VERIFIED.
Premise P_exchange_form is REFUTED: unconditional exchange AND simple ordering predicates (`edge_target_quantity < edge_source_quantity` and `edge_source_quantity < edge_target_quantity`) have ALREADY been exhaustively refuted on paid transitions — do NOT re-propose them!

In directional pairwise transport (`direction` in {"right", "left"}), each directed edge (source -> target) within `0 < sign*(x_target - x_source) < radius` transfers `gamma * q_source` from source to target ONLY WHEN `edge_predicate` evaluates to non-zero (true).
Use `necessary_active_transfer_geometry` (which lists certified in-radius edges with `fired: true` vs `fired: false` and their exact `(q_src, q_dst, x_src, x_dst)`) plus any `counterexamples` in `previous_gate_feedback` to infer the exact conjunctive/algebraic condition over `edge_source_quantity`, `edge_target_quantity`, `edge_source_position`, `edge_target_position` (for example sign requirements, difference thresholds, or compound `and`/`or` conditions). Propose 3 to 4 distinct candidate predicates per call.

Program JSON node format (every node MUST use this schema):
{"op": <op>, "value": <float|null>, "name": <str|null>, "lag": 0, "children": [...]}
- Constant leaf: {"op": "const", "value": 0.1, "name": null, "lag": 0, "children": []}
- Lexical argument leaves:
  * {"op": "arg", "value": null, "name": "edge_source_quantity", "lag": 0, "children": []}
  * {"op": "arg", "value": null, "name": "edge_target_quantity", "lag": 0, "children": []}
  * {"op": "arg", "value": null, "name": "edge_source_position", "lag": 0, "children": []}
  * {"op": "arg", "value": null, "name": "edge_target_position", "lag": 0, "children": []}
- Unary ops (1 child): "abs", "not"
- Binary ops (2 children [left, right]): "add", "sub", "mul", "min", "max", "lt" (left < right), "ge" (left >= right), "and", "or"
- Ternary op (3 children [cond, then, else]): "if"

Return ONLY valid JSON matching (propose up to 4 candidate predicates):
{"candidates": [{"rationale": "...", "directions": ["right", "left"], "edge_predicate": <Program_AST>}]}"""


_READOUT_BASIS_INSTRUCTIONS = """You are the open-grammar hypothesis author inside Hegel Machine v10, gated by PremiseDependencyGraph.
All state-transition and spatial premises (P_role_partition, P_conservation, P_exchange_form, P_interaction_direction, P_position_ownership, P_locality_radius, P_focus_membership) are EMPIRICALLY VERIFIED!
Leaf readout / state-sufficiency premise P_temporal_readout is REFUTED: both the Markovian linear basis AND linear reservoirs in `(q, u)` leave a structured residual (`parent_training_loss > 0.01`).

Important:
- Inspect `trajectory_step` (`0` = fresh reset, `1, 2, ...` = consecutive history steps along a probe) in `role_projected_residual_table`:
  1. If `ols_residual` depends purely on current step `(focused_q, focused_u, focused_x)` regardless of `trajectory_step`, propose an INSTANTANEOUS nonlinear role-uniform term (e.g., `f_i * poly(q_i, u_i, x_i)`).
  2. If `ols_residual` builds up across `trajectory_step = 0, 1, 2, ...` with exponential memory (a POMDP hidden reservoir per object `z_{i,t} = alpha * z_{i,t-1} + drive(q_{i,t}, u_{i,t}, x_{i,t})` read out at the focused object `f_i * z_{i,t}` or summed globally), wrap the nonlinear per-object `drive` inside `accumulate`! (Note: the symbolic gate automatically optimizes the continuous decay pole `value` via 1D golden-section search, so an approximate initial pole like `0.65` is sufficient!)
- Always hedge across your 4 candidates in a single call by including BOTH instantaneous nonlinear terms AND per-object `accumulate(0.65, <nonlinear_drive>, 0.0)` reservoir versions when `trajectory_step >= 1` rows are present! Also check `candidate_reset_step0_only_mse` vs `candidate_training_loss` in `previous_gate_feedback`: if `candidate_reset_step0_only_mse` is ~0 while `candidate_training_loss > 1e-4`, your instantaneous drive is exact at `t=0` and MUST be wrapped inside `accumulate(0.65, <drive>, 0.0)`!
- Note that `parent_ols_pred` was refitted by OLS on `target`, so `ols_residual = target - parent_ols_pred` is mean-zero (`c * h` minus its linear projection). Compare `trajectory_step == 0` rows vs `trajectory_step >= 1` rows!

Each role_term is either:
1. {"kind": "all_object_sum", "summand": <Program_AST over features "object_role:q", "object_role:u", "object_role:x", "object_role:f">}
   - Instantaneous neutral syntax example `sum_i (u_i * x_i)`:
     {"kind": "all_object_sum", "summand": {"op": "mul", "value": null, "name": null, "lag": 0, "children": [{"op": "feature", "value": null, "name": "object_role:u", "lag": 0, "children": []}, {"op": "feature", "value": null, "name": "object_role:x", "lag": 0, "children": []}]}}
   - Recurrent per-object reservoir neutral syntax example `sum_i (f_i * accumulate(0.65, abs(u_i), 0.0))`:
     {"kind": "all_object_sum", "summand": {"op": "mul", "value": null, "name": null, "lag": 0, "children": [{"op": "feature", "value": null, "name": "object_role:f", "lag": 0, "children": []}, {"op": "accumulate", "value": 0.65, "name": null, "lag": 0, "children": [{"op": "abs", "value": null, "name": null, "lag": 0, "children": [{"op": "feature", "value": null, "name": "object_role:u", "lag": 0, "children": []}]}, {"op": "const", "value": 0.0, "name": null, "lag": 0, "children": []}]}]}}
2. A temporal wrapper around (1):
   {"kind": "program_op", "op": "window_sum" | "event_delta", "lag": <int 1..4>, "children": [<role_term>]}

Propose up to 4 diverse candidates per call. Return ONLY valid JSON matching:
{"candidates": [{"rationale": "...", "extra_role_terms": [<role_term>, ...]}]}"""


def propose_flat_unconstrained_programs(
    spec,
    training: list[Observation],
    validation: list[Observation],
    raw_transitions: list[dict],
    budget: SearchBudget,
    transport: Callable[..., tuple[dict | None, dict]],
    *,
    incumbent_loss: float | None = None,
    max_llm_calls: int = 2,
    tolerance: float = 1e-7,
    model_name: str = "gpt-5.6-sol",
) -> dict:
    """Single-factor baseline: LLM proposes flat ``Program`` ASTs without premise graph."""
    start = budget.spent
    ledger = empty_llm_ledger(model=model_name, backend="responses_http")
    allowed_features = set(spec.feature_names)
    fitter = _Fitter(list(training), budget, spec.risk_quantile)
    fallback = initial_prediction(training, spec.risk_quantile)

    sample_obs = [
        {
            "step": idx,
            "features": {k: round(float(v), 5) for k, v in row.features.items()},
            "target": round(float(row.target), 6),
            "history_len": len(row.history),
        }
        for idx, row in enumerate(training[:16])
    ]
    sample_tr = [
        {
            "before": {k: round(float(v), 5) for k, v in tr["before"].items()},
            "after": {k: round(float(v), 5) for k, v in tr["after"].items()},
        }
        for tr in raw_transitions[:16]
        if "before" in tr and "after" in tr
    ]

    evaluated_entries: list[dict] = []
    best_entry: dict | None = None
    feedback_history: list[dict] = []

    for call_idx in range(max_llm_calls):
        prompt_payload = {
            "attempt": call_idx + 1,
            "feature_names": list(spec.feature_names),
            "incumbent_flat_mse": incumbent_loss,
            "sample_observations": sample_obs,
            "sample_transitions": sample_tr,
            "previous_attempt_feedback": feedback_history,
        }
        parsed, usage = transport(
            _FLAT_AUTHOR_INSTRUCTIONS,
            prompt_payload,
            purpose=f"flat_unconstrained_attempt_{call_idx + 1}",
        )
        _record_usage(ledger, usage)
        if not parsed:
            feedback_history.append({"attempt": call_idx + 1, "error": usage.get("error", "invalid_json")})
            continue

        raw_hyps = parsed.get("hypotheses") or ([parsed] if "basis_terms" in parsed else ())
        for hyp_idx, hyp in enumerate(raw_hyps[:4]):
            raw_terms = hyp.get("basis_terms") or []
            terms = []
            for raw_t in raw_terms[:8]:
                if isinstance(raw_t, dict):
                    prog = validate_flat_program_ast(raw_t, allowed_features)
                    if prog is not None:
                        terms.append(prog)
            if not terms:
                continue
            fit = fitter.fit(tuple(terms))
            if fit is None:
                break
            val_loss = (
                loss(fit["program"], validation, fallback, spec.risk_quantile)
                if validation and budget.spend(len(validation))
                else float(fit["loss"])
            )
            cand = Candidate(
                f"flat_llm:{call_idx}:{hyp_idx}",
                fit["program"],
                "flat_llm_unconstrained",
                str(hyp.get("rationale", ""))[:160],
            )
            entry = {
                "candidate": cand,
                "kind": "flat_llm_hypothesis",
                "family_id": "T_OPEN_FLAT",
                "training_loss": float(fit["loss"]),
                "validation_loss": float(val_loss),
                "terms_count": len(terms),
            }
            evaluated_entries.append(entry)
            if best_entry is None or entry["validation_loss"] < best_entry["validation_loss"]:
                best_entry = entry

        feedback_history.append(
            {
                "attempt": call_idx + 1,
                "best_training_loss": best_entry["training_loss"] if best_entry else None,
                "best_validation_loss": best_entry["validation_loss"] if best_entry else None,
            }
        )
        if best_entry is not None and best_entry["validation_loss"] <= tolerance:
            break

    return {
        "mode": "llm_without_premise_gate",
        "supported": bool(best_entry and best_entry["validation_loss"] <= tolerance),
        "best_entry": best_entry,
        "evaluated_count": len(evaluated_entries),
        "compute_cost": budget.spent - start,
        "llm_ledger": ledger,
    }


def propose_premise_gated_commitment_revisions(
    commitment: dict,
    model: dict,
    records: list[dict],
    spec,
    training: list[Observation],
    validation: list[Observation],
    budget: SearchBudget,
    transport: Callable[..., tuple[dict | None, dict]],
    *,
    audit_report: dict | None = None,
    max_llm_calls: int = 2,
    tolerance: float = 1e-7,
    model_name: str = "gpt-5.6-sol",
) -> dict:
    """Single-factor v10 treatment: PremiseDependencyGraph localizes violation & gates LLM ASTs."""
    start = budget.spent
    ledger = empty_llm_ledger(model=model_name, backend="responses_http")
    audit = audit_report or audit_transition_premises(
        model, records, budget, commitments=[commitment], observations=training, tolerance=tolerance
    )
    refuted_ids = [item.get("premise_id") for item in audit.get("refuted_premises", ())]
    fallback = initial_prediction(training, spec.risk_quantile)

    # Branch 1: P_exchange_form refuted on transitions while P_role_partition & P_conservation hold!
    if "P_exchange_form" in refuted_ids and audit.get("conservation_supported"):
        payload_data = audit.get("transition_payload") or {}
        q_rows = payload_data.get("q_rows") or []
        x_rows = payload_data.get("x_rows") or []
        delta_rows = payload_data.get("delta_rows") or []
        dynamic = payload_data.get("dynamic") or list(model.get("dynamic_fields", ()))
        x_fields = payload_data.get("x_fields") or list((model.get("position_candidates") or {}).keys())
        eff_tol = float(payload_data.get("eff_tol", max(1e-12, tolerance)))
        bijections, _ = _enumerate_position_bijections(
            dynamic, x_fields, model["position_candidates"]
        )
        nec_geom = extract_necessary_active_transfer_geometry(
            q_rows, x_rows, delta_rows, bijections, budget, eff_tol
        )
        active_indices = [
            tau for tau in range(len(q_rows)) if max(abs(v) for v in delta_rows[tau]) > 1e-9
        ]
        inactive_indices = [
            tau
            for tau in range(len(q_rows))
            if max(abs(v) for v in delta_rows[tau]) <= 1e-9
            and max(q_rows[tau]) - min(q_rows[tau]) > 0.05
            and max(q_rows[tau]) > 0.05
        ]
        chosen_taus = sorted([*active_indices[-14:], *inactive_indices[-8:]])
        sample_transitions = [
            {
                "tau": tau,
                "active": bool(max(abs(v) for v in delta_rows[tau]) > 1e-9),
                "q": [round(q_rows[tau][i], 5) for i in range(len(dynamic))],
                "unpaired_x_sensors": [round(x_rows[tau][j], 5) for j in range(len(x_fields))],
                "delta_transport": [round(delta_rows[tau][i], 6) for i in range(len(dynamic))],
            }
            for tau in chosen_taus
        ]
        already_refuted = [
            "unconditional (always true)",
            "lt(arg(edge_target_quantity), arg(edge_source_quantity))",
            "lt(arg(edge_source_quantity), arg(edge_target_quantity))",
        ]

        feedback: list[dict] = []
        verified_entries: list[dict] = []
        best_entry: dict | None = None
        updated_model = deepcopy(model)

        for call_idx in range(max_llm_calls):
            prompt_payload = {
                "attempt": call_idx + 1,
                "isolated_violated_premise": "P_exchange_form",
                "verified_upstream_premises": audit.get("supported_premises", []),
                "already_refuted_predicates": already_refuted,
                "committed_direction": commitment["radius_parameters"][0]["direction"]
                if commitment.get("radius_parameters")
                else None,
                "object_count": len(dynamic),
                "necessary_active_transfer_geometry": nec_geom,
                "raw_sample_transitions": sample_transitions,
                "previous_gate_feedback": feedback,
            }
            parsed, usage = transport(
                _EDGE_PREDICATE_INSTRUCTIONS,
                prompt_payload,
                purpose=f"premise_gated_edge_predicate_{call_idx + 1}",
            )
            _record_usage(ledger, usage)
            if not parsed:
                feedback.append({"attempt": call_idx + 1, "rejected": usage.get("error", "invalid_json")})
                continue

            raw_cands = parsed.get("candidates") or ([parsed] if "edge_predicate" in parsed else ())
            for cand_spec in raw_cands[:4]:
                raw_pred = cand_spec.get("edge_predicate")
                if not isinstance(raw_pred, dict):
                    continue
                pred_prog = validate_edge_predicate_ast(raw_pred)
                if pred_prog is None:
                    feedback.append({"attempt": call_idx + 1, "rejected": "invalid_edge_predicate_ast"})
                    continue
                dirs = tuple(
                    d for d in (cand_spec.get("directions") or ["right", "left"]) if d in {"right", "left"}
                ) or ("right", "left")
                screened = screen_custom_edge_predicate_on_transitions(
                    q_rows,
                    x_rows,
                    delta_rows,
                    dynamic,
                    x_fields,
                    bijections,
                    pred_prog,
                    dirs,
                    budget=budget,
                    eff_tol=eff_tol,
                )
                if not screened["supported"]:
                    fb_entry: dict = {
                        "attempt": call_idx + 1,
                        "predicate": pred_prog.describe(),
                        "rejected": "failed_transition_premise_verification",
                    }
                    if screened.get("counterexamples"):
                        fb_entry["counterexamples"] = screened["counterexamples"]
                    feedback.append(fb_entry)
                    continue

                verified_pred_prog = screened.get("calibrated_predicate") or pred_prog
                # Custom predicate passed exact transition verification! Narrow x->q and f->q!
                matches_by_dir: dict[str, list[dict]] = {}
                for m in screened["matches"]:
                    matches_by_dir.setdefault(m["direction"], []).append(m)
                narrowed_pos = {
                    xf: sorted({m["mapping"][xf] for m in screened["matches"]}) for xf in x_fields
                }
                focus_audit = _audit_focus_membership_from_trajectories(
                    model,
                    records,
                    narrowed_pos,
                    budget,
                    eff_tol=eff_tol,
                )
                narrowed_focus = (
                    focus_audit["narrowed_focus_domains"]
                    if focus_audit is not None and focus_audit.get("narrowed_focus_domains")
                    else None
                )
                synth_audit = {
                    **deepcopy(audit),
                    "status": "verified_conditional_exchange",
                    "exhaustive_bijections": True,
                    "matches_by_direction": matches_by_dir,
                    "narrowed_position_domains": narrowed_pos,
                    "narrowed_focus_domains": narrowed_focus,
                    "verified_direction": next(iter(matches_by_dir)) if len(matches_by_dir) == 1 else None,
                    "verified_edge_condition": verified_pred_prog.to_dict(),
                }
                updated_model = apply_verified_transition_premises(model, synth_audit)
                schema_dict = updated_model.get("schema")
                if not schema_dict:
                    from .object_alternatives import enumerate_object_schemas

                    enum_res = enumerate_object_schemas(updated_model, budget, max_schemas=8)
                    candidate_schemas = enum_res["schemas"]
                else:
                    candidate_schemas = [schema_dict]

                sym = commitment["radius_parameters"][0]["symbol"]
                committed_dir = commitment["radius_parameters"][0]["direction"]
                committed_cond = commitment["radius_parameters"][0].get("edge_condition", "unconditional")
                for sc in candidate_schemas:
                    sc_obj = _as_schema(sc)
                    if sc_obj is None:
                        continue
                    sc_x_to_q = {
                        sc_obj.fields[g]["x"]: sc_obj.fields[g]["q"] for g in sc_obj.objects
                    }
                    for v_dir, dir_matches in matches_by_dir.items():
                        compat = [m for m in dir_matches if m["mapping"] == sc_x_to_q]
                        if not compat:
                            continue
                        r_bounds = (
                            min(m["radius_interval"][0] for m in compat),
                            max(m["radius_interval"][1] for m in compat),
                        )
                        rev_meta = {
                            "trigger_source": "premise_gated_open_grammar_author",
                            "revision_kind": "open_grammar_exchange_condition_revision",
                            "parent_commitment_id": commitment["commitment_id"],
                            "author_model": model_name,
                            "retracted_premises": [
                                {
                                    "premise_id": f"exchange_form:{sym}",
                                    "edge_condition": committed_cond,
                                    "counterexample": deepcopy(
                                        audit["refuted_premises"][0].get("counterexample")
                                    ),
                                }
                            ],
                            "replacement_premises": [
                                {
                                    "premise_id": f"exchange_form:{sym}",
                                    "edge_condition": verified_pred_prog.to_dict(),
                                    "direction": v_dir,
                                    "supporting_transitions": len(q_rows),
                                }
                            ],
                            "retained_premises": [
                                p["premise_id"]
                                for p in commitment.get("support_premises", ())
                                if p["premise_id"]
                                not in {f"exchange_form:{sym}", f"interaction_direction:{sym}"}
                            ],
                        }
                        if v_dir != committed_dir:
                            rev_meta["retracted_premises"].append(
                                {"premise_id": f"interaction_direction:{sym}", "direction": committed_dir}
                            )
                            rev_meta["replacement_premises"].append(
                                {"premise_id": f"interaction_direction:{sym}", "direction": v_dir}
                            )
                        entries = instantiate_commitment_on_schema(
                            spec,
                            training,
                            sc_obj,
                            commitment,
                            budget,
                            radius_bounds=r_bounds,
                            direction_overrides={sym: v_dir},
                            predicate_overrides={sym: verified_pred_prog.to_dict()},
                            reorganization_type="scientific_premise_revision",
                            scientific_revision=rev_meta,
                            tolerance=tolerance,
                        )
                        for entry in entries:
                            val_loss = (
                                loss(entry["candidate"].program, validation, fallback, spec.risk_quantile)
                                if validation and budget.spend(len(validation))
                                else entry["training_loss"]
                            )
                            entry["validation_loss"] = float(val_loss)
                            entry["resolved_schema"] = sc_obj.to_dict()
                            verified_entries.append(entry)
                            if best_entry is None or val_loss < best_entry["validation_loss"]:
                                best_entry = entry

            if best_entry is not None and best_entry["validation_loss"] <= tolerance:
                break

        return {
            "mode": "llm_with_v10_premise_gate",
            "isolated_violated_premise": "P_exchange_form",
            "supported": bool(best_entry and best_entry["validation_loss"] <= tolerance),
            "best_entry": best_entry,
            "updated_model": updated_model,
            "verified_candidate_count": len(verified_entries),
            "compute_cost": budget.spent - start,
            "llm_ledger": ledger,
        }

    # Branch 1B: P_latent_transition_drive refuted on transitions (nonlinear POMDP transition reservoir)!
    if "P_latent_transition_drive" in refuted_ids and audit.get("transition_latent_payload"):
        lat_payload = audit["transition_latent_payload"]
        valid_rows_l = lat_payload["valid_rows"]
        dynamic_l = lat_payload["dynamic"]
        drive_by_q_l = lat_payload["drive_by_q"]
        x_fields_l = lat_payload["x_fields"]
        eff_tol_l = float(lat_payload.get("eff_tol", max(1e-12, tolerance)))
        updated_model = apply_verified_transition_premises(model, audit)
        bijections_l, _ = _enumerate_position_bijections(
            dynamic_l, x_fields_l, updated_model.get("position_candidates") or model["position_candidates"]
        )
        schema_dict_l = updated_model.get("schema")
        if not schema_dict_l:
            from .object_alternatives import enumerate_object_schemas

            enum_l = enumerate_object_schemas(updated_model, budget, max_schemas=6)
            cand_schemas_l = enum_l["schemas"]
        else:
            cand_schemas_l = [schema_dict_l]

        feedback_l: list[dict] = []
        verified_entries_l: list[dict] = []
        best_entry_l: dict | None = None
        sym_l = commitment["radius_parameters"][0]["symbol"]

        for call_idx in range(max_llm_calls):
            prompt_payload_l = {
                "attempt": call_idx + 1,
                "isolated_violated_premise": "P_latent_transition_drive",
                "verified_upstream_premises": [
                    "P_role_partition",
                    "P_reset_conservation",
                    "P_interaction_direction",
                    "P_position_ownership",
                ],
                "already_refuted_reservoirs": [
                    "markov_sufficient",
                    "static_sink_and_nonlocal",
                    "first_order_linear_reservoir(u, q)",
                    "spatially_coupled_linear_reservoir(u, q)",
                    "second_order_ar2_linear_reservoir(u, q)",
                ],
                "object_count": len(dynamic_l),
                "verified_direction": lat_payload.get("representative_direction"),
                "deflated_trajectory_samples": lat_payload.get("deflated_trajectory_samples", []),
                "previous_gate_feedback": feedback_l,
            }
            parsed_l, usage_l = transport(
                _LATENT_TRANSITION_DRIVE_INSTRUCTIONS,
                prompt_payload_l,
                purpose=f"premise_gated_latent_transition_drive_{call_idx + 1}",
            )
            _record_usage(ledger, usage_l)
            if not parsed_l:
                feedback_l.append({"attempt": call_idx + 1, "rejected": usage_l.get("error", "invalid_json")})
                continue

            raw_cands_l = parsed_l.get("candidates") or (
                [parsed_l] if ("latent_drive_ast" in parsed_l or "latent_drive" in parsed_l) else ()
            )
            for cand_spec_l in raw_cands_l[:4]:
                raw_drive = cand_spec_l.get("latent_drive_ast") or cand_spec_l.get("latent_drive")
                if not isinstance(raw_drive, dict):
                    continue
                drive_prog = validate_object_drive_ast(raw_drive)
                if drive_prog is None:
                    feedback_l.append({"attempt": call_idx + 1, "rejected": "invalid_latent_drive_ast"})
                    continue
                scr_res = screen_custom_latent_transition_drive(
                    valid_rows_l,
                    dynamic_l,
                    drive_by_q_l,
                    x_fields_l,
                    bijections_l,
                    drive_prog,
                    (lat_payload.get("representative_direction") or "right",),
                    budget=budget,
                    eff_tol=eff_tol_l,
                    observations=training,
                )
                if not scr_res.get("supported"):
                    feedback_l.append(
                        {
                            "attempt": call_idx + 1,
                            "drive_ast": drive_prog.describe(),
                            "rejected": scr_res.get("reason", "failed_transition_deflation"),
                        }
                    )
                    continue

                v_pole = float(scr_res["verified_pole"])
                matches_by_dir_l = scr_res["matches_by_direction"]
                exp_role_terms = expand_role_terms_for_custom_latent_transition(
                    commitment["role_terms"], drive_prog, v_pole
                )
                for sc_l in cand_schemas_l:
                    sc_obj_l = _as_schema(sc_l)
                    if sc_obj_l is None:
                        continue
                    sc_x_q = {
                        sc_obj_l.fields[g]["x"]: sc_obj_l.fields[g]["q"] for g in sc_obj_l.objects
                    }
                    for v_dir_l, d_matches_l in matches_by_dir_l.items():
                        compat_l = [m for m in d_matches_l if m["mapping"] == sc_x_q]
                        if not compat_l:
                            continue
                        r_bounds_l = (
                            min(m["radius_interval"][0] for m in compat_l),
                            max(m["radius_interval"][1] for m in compat_l),
                        )
                        rev_meta_l = {
                            "trigger_source": "premise_gated_open_grammar_author",
                            "revision_kind": "open_grammar_latent_transition_drive_revision",
                            "parent_commitment_id": commitment["commitment_id"],
                            "author_model": model_name,
                            "latent_state_mechanism": "custom_transition_memory",
                            "latent_pole": v_pole,
                            "retracted_premises": [
                                {
                                    "premise_id": "markov_state_sufficiency",
                                    "expected": "supported",
                                },
                                {
                                    "premise_id": "P_latent_transition_drive",
                                    "expected": "linear_u_q",
                                },
                            ],
                            "replacement_premises": [
                                {
                                    "premise_id": "markov_state_sufficiency",
                                    "latent_state_mechanism": "custom_transition_memory",
                                    "latent_pole": v_pole,
                                    "custom_drive": drive_prog.to_dict(),
                                }
                            ],
                            "retained_premises": [
                                p["premise_id"]
                                for p in commitment.get("support_premises", ())
                                if p["premise_id"] != "markov_state_sufficiency"
                            ],
                        }
                        cand_r_intervals = [r_bounds_l]
                        if r_bounds_l[1] - r_bounds_l[0] > 1e-5:
                            r_mid_l = 0.5 * (r_bounds_l[0] + r_bounds_l[1])
                            cand_r_intervals.extend(
                                [(r_bounds_l[0], r_mid_l), (r_mid_l, r_bounds_l[1])]
                            )
                        for sub_rb_l in cand_r_intervals:
                            entries_l = instantiate_commitment_on_schema(
                                spec,
                                training,
                                sc_obj_l,
                                commitment,
                                budget,
                                radius_bounds=sub_rb_l,
                                direction_overrides={sym_l: v_dir_l},
                                role_terms_override=exp_role_terms,
                                latent_state_mechanism="custom_transition_memory",
                                reorganization_type="scientific_premise_revision",
                                scientific_revision=rev_meta_l,
                                tolerance=tolerance,
                            )
                            hit_tol_l = False
                            for entry_l in entries_l:
                                val_loss_l = (
                                    loss(entry_l["candidate"].program, validation, fallback, spec.risk_quantile)
                                    if validation and budget.spend(len(validation))
                                    else entry_l["training_loss"]
                                )
                                entry_l["validation_loss"] = float(val_loss_l)
                                entry_l["resolved_schema"] = sc_obj_l.to_dict()
                                verified_entries_l.append(entry_l)
                                if best_entry_l is None or val_loss_l < best_entry_l["validation_loss"]:
                                    best_entry_l = entry_l
                                if val_loss_l <= tolerance:
                                    hit_tol_l = True
                            if hit_tol_l:
                                break

            if best_entry_l is not None and best_entry_l["validation_loss"] <= tolerance:
                break

        return {
            "mode": "llm_with_v10_premise_gate",
            "isolated_violated_premise": "P_latent_transition_drive",
            "supported": bool(best_entry_l and best_entry_l["validation_loss"] <= tolerance),
            "best_entry": best_entry_l,
            "updated_model": updated_model,
            "verified_candidate_count": len(verified_entries_l),
            "compute_cost": budget.spent - start,
            "llm_ledger": ledger,
        }

    # Branch 2: Upstream transition & geometry premises verified -> P_temporal_readout refuted!
    updated_model = apply_verified_transition_premises(model, audit)
    schema_dict = updated_model.get("schema")
    if not schema_dict:
        from .object_alternatives import enumerate_object_schemas

        enum_res = enumerate_object_schemas(updated_model, budget, max_schemas=4)
        candidate_schemas = enum_res["schemas"]
    else:
        candidate_schemas = [schema_dict]
    if not candidate_schemas:
        return {
            "mode": "llm_with_v10_premise_gate",
            "isolated_violated_premise": "unresolved_schema",
            "supported": False,
            "best_entry": None,
            "updated_model": updated_model,
            "verified_candidate_count": 0,
            "compute_cost": budget.spent - start,
            "llm_ledger": ledger,
        }

    primary_schema = _as_schema(candidate_schemas[0])
    sc_x_to_q = {
        primary_schema.fields[g]["x"]: primary_schema.fields[g]["q"] for g in primary_schema.objects
    }
    sym = commitment["radius_parameters"][0]["symbol"] if commitment.get("radius_parameters") else None
    verified_dir = audit.get("verified_direction") or (
        commitment["radius_parameters"][0]["direction"] if sym else "symmetric"
    )
    dir_overrides = {sym: verified_dir} if sym else {}
    compat_matches = [
        m
        for m in (audit.get("matches_by_direction") or {}).get(verified_dir, ())
        if m.get("mapping") == sc_x_to_q
    ]
    if compat_matches:
        r_bounds = (
            min(m["radius_interval"][0] for m in compat_matches),
            max(m["radius_interval"][1] for m in compat_matches),
        )
    elif audit.get("radius_interval"):
        r_bounds = (
            float(audit["radius_interval"][0]),
            float(audit["radius_interval"][1]),
        )
    else:
        r_bounds = None

    parent_entries = instantiate_commitment_on_schema(
        spec,
        training,
        primary_schema,
        commitment,
        budget,
        radius_bounds=r_bounds,
        direction_overrides=dir_overrides,
        tolerance=tolerance,
    )
    parent_entry = parent_entries[0] if parent_entries else None
    parent_prog = parent_entry["candidate"].program if parent_entry else Program("const", value=0.0)
    parent_loss = parent_entry["training_loss"] if parent_entry else float("inf")
    existing_role_term_summaries = [
        json.dumps(rt, sort_keys=True)[:220] for rt in commitment.get("role_terms", ())
    ]

    # Prefer a contiguous multi-step trajectory slice (history_len = 0, 1, 2, ...) so the
    # author can distinguish instantaneous nonlinearities from POMDP reservoir memory!
    selected_rows: list[Observation] = []
    best_contig: list[Observation] = []
    cur_contig: list[Observation] = []
    for r_obs in training:
        hlen = len(r_obs.history)
        if hlen == 0:
            if len(cur_contig) > len(best_contig):
                best_contig = cur_contig
            cur_contig = [r_obs]
        elif len(cur_contig) == hlen:
            cur_contig.append(r_obs)
        else:
            if len(cur_contig) > len(best_contig):
                best_contig = cur_contig
            cur_contig = []
    if len(cur_contig) > len(best_contig):
        best_contig = cur_contig
    if len(best_contig) >= 6:
        selected_rows.extend(best_contig[:16])
        reset_extras = [r_obs for r_obs in training if len(r_obs.history) == 0 and r_obs not in selected_rows]
        selected_rows.extend(reset_extras[:6])
    else:
        selected_rows = list(training[:20])

    residual_samples = []
    for idx, row in enumerate(selected_rows):
        pred = parent_prog.predict(row)
        res_val = float(row.target - pred)
        fq = sum(
            row.features[primary_schema.fields[g]["f"]] * row.features[primary_schema.fields[g]["q"]]
            for g in primary_schema.objects
        )
        fu = sum(
            row.features[primary_schema.fields[g]["f"]] * row.features[primary_schema.fields[g]["u"]]
            for g in primary_schema.objects
        )
        fx = sum(
            row.features[primary_schema.fields[g]["f"]] * row.features[primary_schema.fields[g]["x"]]
            for g in primary_schema.objects
        )
        tq = sum(row.features[primary_schema.fields[g]["q"]] for g in primary_schema.objects)
        tu = sum(row.features[primary_schema.fields[g]["u"]] for g in primary_schema.objects)
        active_obj = next(
            (
                o_idx
                for o_idx, g in enumerate(primary_schema.objects)
                if abs(row.features[primary_schema.fields[g]["f"]] - 1.0) <= 1e-6
            ),
            None,
        )
        residual_samples.append(
            {
                "row_index": idx,
                "trajectory_step": len(row.history),
                "active_focus_object": active_obj,
                "per_object_q": [
                    round(float(row.features[primary_schema.fields[g]["q"]]), 4)
                    for g in primary_schema.objects
                ],
                "per_object_u": [
                    round(float(row.features[primary_schema.fields[g]["u"]]), 4)
                    for g in primary_schema.objects
                ],
                "focused_q": round(float(fq), 5),
                "focused_u": round(float(fu), 5),
                "focused_x": round(float(fx), 5),
                "total_q": round(float(tq), 5),
                "total_u": round(float(tu), 5),
                "target": round(float(row.target), 6),
                "parent_ols_pred": round(float(pred), 6),
                "ols_residual": round(res_val, 6),
            }
        )

    feedback = []
    verified_entries = []
    best_entry = None

    for call_idx in range(max_llm_calls):
        prompt_payload = {
            "attempt": call_idx + 1,
            "isolated_violated_premise": "P_temporal_readout",
            "verified_upstream_premises": audit.get("supported_premises", []),
            "parent_training_loss": parent_loss,
            "existing_role_terms": existing_role_term_summaries,
            "role_projected_residual_table": residual_samples,
            "previous_gate_feedback": feedback,
        }
        parsed, usage = transport(
            _READOUT_BASIS_INSTRUCTIONS,
            prompt_payload,
            purpose=f"premise_gated_readout_basis_{call_idx + 1}",
        )
        _record_usage(ledger, usage)
        if not parsed:
            feedback.append({"attempt": call_idx + 1, "rejected": usage.get("error", "invalid_json")})
            continue

        raw_cands = parsed.get("candidates") or ([parsed] if "extra_role_terms" in parsed else ())
        for cand_spec in raw_cands[:4]:
            raw_extras = cand_spec.get("extra_role_terms") or []
            valid_extras = []
            for item in raw_extras[:4]:
                norm = validate_role_term_spec(item, primary_schema)
                if norm is not None:
                    valid_extras.append(norm)
            if not valid_extras:
                feedback.append({"attempt": call_idx + 1, "rejected": "invalid_or_non_role_uniform_terms"})
                continue

            dedup_combined = []
            seen_pole_norm = set()
            for rt_c in [*deepcopy(commitment["role_terms"]), *valid_extras]:
                norm_key = json.dumps(
                    _rebind_latent_pole_in_role_terms([rt_c], "__LATENT_POLE__")[0],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if norm_key not in seen_pole_norm:
                    seen_pole_norm.add(norm_key)
                    dedup_combined.append(rt_c)
            combined_role_terms = dedup_combined
            extra_poles = _extract_latent_poles_from_role_terms(valid_extras)
            eff_lat_mech = "custom_focal_memory" if extra_poles else commitment.get("latent_state_mechanism")
            cand_best_train = None
            cand_best_val = None
            for sc in candidate_schemas:
                sc_obj = _as_schema(sc)
                if sc_obj is None:
                    continue
                rev_meta = {
                    "trigger_source": "premise_gated_open_grammar_author",
                    "revision_kind": "open_grammar_temporal_readout_revision",
                    "parent_commitment_id": commitment["commitment_id"],
                    "author_model": model_name,
                    "latent_state_mechanism": eff_lat_mech,
                    "retracted_premises": [
                        {
                            "premise_id": "temporal_readout_basis",
                            "parent_training_loss": float(parent_loss),
                        }
                    ],
                    "replacement_premises": [
                        {
                            "premise_id": "temporal_readout_basis",
                            "added_role_terms": deepcopy(valid_extras),
                        }
                    ],
                    "retained_premises": [
                        p["premise_id"]
                        for p in commitment.get("support_premises", ())
                        if p["premise_id"] != "temporal_readout_basis"
                    ],
                }
                cand_r_b2 = [r_bounds]
                if r_bounds is not None and r_bounds[1] - r_bounds[0] > 1e-5:
                    rm2 = 0.5 * (float(r_bounds[0]) + float(r_bounds[1]))
                    cand_r_b2.extend([(float(r_bounds[0]), rm2), (rm2, float(r_bounds[1]))])
                for sub_rb2 in cand_r_b2:
                    entries = instantiate_commitment_on_schema(
                        spec,
                        training,
                        sc_obj,
                        commitment,
                        budget,
                        radius_bounds=sub_rb2,
                        direction_overrides=dir_overrides,
                        role_terms_override=combined_role_terms,
                        latent_state_mechanism=eff_lat_mech,
                        reorganization_type="scientific_premise_revision",
                        scientific_revision=rev_meta,
                        tolerance=tolerance,
                    )
                    hit_tol2 = False
                    for entry in entries:
                        val_loss = (
                            loss(entry["candidate"].program, validation, fallback, spec.risk_quantile)
                            if validation and budget.spend(len(validation))
                            else entry["training_loss"]
                        )
                        entry["validation_loss"] = float(val_loss)
                        entry["resolved_schema"] = sc_obj.to_dict()
                        verified_entries.append(entry)
                        if cand_best_val is None or val_loss < cand_best_val:
                            cand_best_val = float(val_loss)
                            cand_best_train = float(entry["training_loss"])
                        if best_entry is None or val_loss < best_entry["validation_loss"]:
                            best_entry = entry
                        if val_loss <= tolerance:
                            hit_tol2 = True
                    if hit_tol2:
                        break

            step0_mse = None
            step0_rows = [
                Observation(dict(r_rec["before"]), float(r_rec["target"]), ())
                for r_rec in records
                if int(r_rec.get("position", 0)) == 0
                and not r_rec.get("history")
                and "before" in r_rec
                and "target" in r_rec
                and set(primary_schema.feature_names) <= set(r_rec["before"])
            ]
            if len(step0_rows) >= 8:
                r_s0 = (
                    0.5 * (float(r_bounds[0]) + float(r_bounds[1]))
                    if r_bounds is not None
                    else float(commitment["radius_parameters"][0]["source_values"][-1])
                    if commitment.get("radius_parameters")
                    else 1.0
                )
                s0_terms = instantiate_commitment_terms(
                    commitment,
                    primary_schema,
                    radius_bindings={sym: r_s0} if sym else None,
                    direction_overrides=dir_overrides,
                    role_terms_override=combined_role_terms,
                )
                if s0_terms is not None:
                    s0_fitter = _Fitter(step0_rows, budget, spec.risk_quantile)
                    s0_fit = s0_fitter.fit(s0_terms)
                    if s0_fit is not None:
                        step0_mse = float(s0_fit["loss"])

            feedback.append(
                {
                    "attempt": call_idx + 1,
                    "tested_extra_role_terms": valid_extras,
                    "candidate_reset_step0_only_mse": step0_mse,
                    "candidate_training_loss": cand_best_train,
                    "candidate_validation_loss": cand_best_val,
                }
            )
            if best_entry is not None and best_entry["validation_loss"] <= tolerance:
                break
        if best_entry is not None and best_entry["validation_loss"] <= tolerance:
            break

    return {
        "mode": "llm_with_v10_premise_gate",
        "isolated_violated_premise": "P_temporal_readout",
        "supported": bool(best_entry and best_entry["validation_loss"] <= tolerance),
        "best_entry": best_entry,
        "updated_model": updated_model,
        "verified_candidate_count": len(verified_entries),
        "compute_cost": budget.spent - start,
        "llm_ledger": ledger,
    }
