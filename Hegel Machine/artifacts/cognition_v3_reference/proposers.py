"""Observed-data-only program proposals from the locally configured Codex CLI.

The CLI is a model transport here: one prompt produces ASTs and optional probe
trajectories. It runs in an ephemeral, read-only temporary directory with its
external tools disabled. The user's configured model and login are inherited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from time import monotonic
from typing import Any, Sequence

from .evaluation import predict
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget, TaskSpec


@dataclass
class ProposalBatch:
    candidates: list[Candidate]
    probes: list[Probe]
    usage: dict[str, Any]


_OPERATORS = {
    "const": 0, "feature": 0, "lag": 0, "available": 0, "abs": 1, "not": 1,
    "add": 2, "sub": 2, "mul": 2, "min": 2, "max": 2,
    "ge": 2, "lt": 2, "and": 2, "or": 2, "if": 3,
}
# Each of these capabilities is available in the inspected Codex 0.153.3 CLI.
# Disabling tools is part of giving this baseline only the common task input.
_DISABLED_FEATURES = (
    "shell_tool", "apps", "plugins", "memories", "multi_agent",
    "code_mode_host", "browser_use", "computer_use", "image_generation",
    "skill_search", "hooks", "view_image",
)

_SYSTEM_INSTRUCTIONS = (
    "You are a bounded program proposal component. Infer executable predictive "
    "expressions only from the supplied observations and public capabilities. "
    "Return the requested JSON schema. Do not use tools, inspect files, browse, "
    "delegate, or infer hidden task labels. A short unsupported guess is worse "
    "than returning no candidate. Keep the complete response concise."
)


def proposal_schema(spec: TaskSpec, max_candidates: int, max_probes: int) -> dict:
    point = {
        "type": "object",
        "properties": {name: {"type": "number"} for name in spec.feature_names},
        "required": list(spec.feature_names), "additionalProperties": False,
    }
    program = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "op": {"type": "string", "enum": list(_OPERATORS)},
            "value": {"type": ["number", "null"]},
            "name": {"type": ["string", "null"]},
            "lag": {"type": "integer"},
            "children": {"type": "array", "items": {"$ref": "#/$defs/program"}},
        },
        "required": ["op", "value", "name", "lag", "children"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "$defs": {"program": program},
        "properties": {
            "candidates": {
                "type": "array", "maxItems": max_candidates,
                "items": {"type": "object", "additionalProperties": False,
                          "properties": {"program": {"$ref": "#/$defs/program"},
                                         "rationale": {"type": "string"}},
                          "required": ["program", "rationale"]},
            },
            "probes": {
                "type": "array", "maxItems": max_probes,
                "items": {"type": "object", "additionalProperties": False,
                          "properties": {
                              "sequence": {"type": "array", "minItems": 1,
                                           "maxItems": spec.max_probe_length, "items": point},
                              "rationale": {"type": "string"}},
                          "required": ["sequence", "rationale"]},
            },
        },
        "required": ["candidates", "probes"],
    }


def build_prompt(observations: Sequence[Observation], incumbent: Program,
                 spec: TaskSpec, budget: SearchBudget, *, max_candidates: int = 8,
                 max_probes: int = 2, max_history: int = 4,
                 remaining_tokens: int | None = None) -> str:
    """Whitelist measured data and public capabilities; omit task metadata."""
    payload = {
        "training_observations": [asdict(row) for row in observations],
        "incumbent": incumbent.to_dict(),
        "capabilities": {
            "feature_names": list(spec.feature_names),
            "feature_bounds": spec.feature_bounds,
            "max_probe_length": spec.max_probe_length,
            "available_probe_sequences": [list(p.sequence) for p in spec.available_probes],
            "max_history": max_history,
        },
        "fixed_objective": {
            "prediction_loss": "squared_error" if spec.risk_quantile == 0.5 else "pinball",
            "risk_quantile": spec.risk_quantile,
            "decision_costs": spec.decision_costs,
        },
        "budget": {"remaining_compute_units": budget.remaining,
                   "remaining_reported_token_budget": remaining_tokens,
                   "max_candidates": max_candidates, "max_probes": max_probes},
        "operator_arities": _OPERATORS,
    }
    return (
        "You are a program proposal component, not a coding agent. Use only the JSON "
        "input below. Do not call tools, read files, access the network, edit source, or "
        "delegate work. Return only the requested JSON object.\n"
        "Propose a few short executable predictive programs, improving or replacing the "
        "incumbent from the supplied training observations. You may also suggest short "
        "probe sequences to distinguish competing explanations; the controller will "
        "execute them and account for observations. You do not receive evaluation data.\n"
        "Program nodes have op, value, name, lag, children. const uses a finite value; "
        "feature uses a supplied current feature name. lag(name,k) reads the kth most "
        "recent historical row, where k>=1; __target__ is available only through lag "
        "when present in supplied histories. Unused value/name are null and unused lag "
        "is 0. available(k) returns whether history contains at least k rows and uses lag=k. "
        "An if with this condition can execute a learned startup model before history exists. "
        "Comparisons return 0 or 1, Boolean operators use zero/nonzero, and if has "
        "condition/true/false children. Missing initial history uses the controller's "
        "shared fitted fallback. All probe coordinates must obey supplied bounds. "
        "An empty proposals list is valid when no useful change is supported. "
        "Keep rationale concise and output within the indicated remaining budget.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )


def _run(command: list[str], prompt: str, timeout: float, cwd: str) -> tuple[int, str, str, bool]:
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=cwd,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr, True


def _events(stdout: str) -> tuple[list[dict], list[str]]:
    events, other_lines = [], []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            other_lines.append(line)
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, other_lines


def _parse_proposals(payload: dict, observations: Sequence[Observation], spec: TaskSpec,
                     budget: SearchBudget, *, call_number: int, max_candidates: int,
                     max_probes: int, max_history: int) -> tuple[list[Candidate], list[Probe], list[str]]:
    candidates, probes, errors = [], [], []
    historical_names = {name for row in observations for entry in row.history for name in entry}
    current_names = set(spec.feature_names)
    for index, row in enumerate(payload.get("candidates", [])[:max_candidates]):
        try:
            program = Program.from_dict(row["program"])
            for name, lag in program.observed_variables():
                if lag == 0 and name not in current_names:
                    raise ValueError(f"unavailable current feature: {name}")
                if lag > 0 and (name not in historical_names or lag > max_history):
                    raise ValueError(f"unavailable historical variable: {name}, lag {lag}")
            if not budget.spend(len(observations)):
                errors.append("compute budget exhausted while validating proposals")
                break
            for observation in observations:
                if not isfinite(predict(program, observation, 0.0)):
                    raise ValueError("non-finite prediction")
            candidates.append(Candidate(f"codex_{call_number}_{index}", program, "llm",
                                        str(row.get("rationale", ""))))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"candidate {index}: {error}")
    for index, row in enumerate(payload.get("probes", [])[:max_probes]):
        try:
            sequence = row["sequence"]
            if not sequence or len(sequence) > spec.max_probe_length:
                raise ValueError("probe length is outside the task capability")
            for point in sequence:
                if set(point) != current_names:
                    raise ValueError("probe must specify exactly the task features")
                for name, value in point.items():
                    if not isinstance(value, (float, int)) or not isfinite(value):
                        raise ValueError("probe coordinates must be finite numbers")
                    if name in spec.feature_bounds:
                        low, high = spec.feature_bounds[name]
                        if not low <= value <= high:
                            raise ValueError(f"probe coordinate outside bounds: {name}")
            probes.append(Probe(f"codex_probe_{call_number}_{index}",
                                tuple({k: float(v) for k, v in point.items()} for point in sequence)))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"probe {index}: {error}")
    return candidates, probes, errors


class CodexProposer:
    """Use configured Codex model/login, with bounded calls and recorded costs.

    token_budget controls admission of subsequent calls from reported usage.
    It is not a backend-enforced per-call token cap; a final call can exceed it.
    """

    def __init__(self, *, executable: str | None = None, timeout_seconds: float = 120,
                 token_budget: int = 12000, max_calls: int = 2, max_candidates: int = 8,
                 max_probes: int = 2, max_history: int = 4):
        self.executable = executable or shutil.which("codex")
        self.timeout_seconds, self.token_budget, self.max_calls = timeout_seconds, token_budget, max_calls
        self.max_candidates, self.max_probes, self.max_history = max_candidates, max_probes, max_history
        self.calls, self.total_tokens = 0, 0
        self.usage_unavailable = False
        self._version: str | None = None

    def propose(self, observations: Sequence[Observation], incumbent: Program,
                spec: TaskSpec, budget: SearchBudget) -> ProposalBatch:
        started, compute_before = monotonic(), budget.spent
        usage: dict[str, Any] = {
            "backend": "codex_cli", "runtime_path": self.executable,
            "runtime_version": self._version, "model": "unreported", "model_selection": "cli_default",
            "input_tokens": None, "cached_input_tokens": None, "output_tokens": None,
            "total_tokens": None, "calls": 0, "wall_time_seconds": 0.0,
            "compute_cost": 0, "status": "not_run", "errors": [],
            "token_budget": self.token_budget, "token_limit_kind": "reported_usage_between_calls",
            "instruction_protocol": "minimal_program_proposer_v2",
            "system_instructions": _SYSTEM_INSTRUCTIONS,
        }
        if not self.executable:
            usage.update(status="runtime_unavailable", errors=["Codex CLI was not found on PATH"])
            return ProposalBatch([], [], usage)
        if self.calls >= self.max_calls or self.total_tokens >= self.token_budget or self.usage_unavailable:
            usage["status"] = "llm_budget_exhausted" if not self.usage_unavailable else "usage_unavailable"
            return ProposalBatch([], [], usage)
        if not observations or budget.remaining < len(observations):
            usage["status"] = "insufficient_observation_or_compute_budget"
            return ProposalBatch([], [], usage)
        prompt = build_prompt(observations, incumbent, spec, budget,
                              max_candidates=self.max_candidates, max_probes=self.max_probes,
                              max_history=self.max_history,
                              remaining_tokens=max(0, self.token_budget - self.total_tokens))
        usage["prompt"] = prompt
        candidates, probes = [], []
        try:
            if self._version is None:
                self._version = subprocess.check_output([self.executable, "--version"],
                                                        text=True, timeout=10).strip()
            usage["runtime_version"] = self._version
            with tempfile.TemporaryDirectory(prefix="hegel-llm-proposer-") as directory:
                schema = Path(directory) / "proposal_schema.json"
                response = Path(directory) / "proposal.json"
                instructions = Path(directory) / "proposer_instructions.txt"
                instructions.write_text(_SYSTEM_INSTRUCTIONS, encoding="utf-8")
                schema.write_text(json.dumps(proposal_schema(spec, self.max_candidates, self.max_probes)), encoding="utf-8")
                command = [self.executable, "-a", "never", "exec", "--sandbox", "read-only",
                           "--cd", directory, "--skip-git-repo-check", "--ephemeral", "--json",
                           "--color", "never", "--output-schema", str(schema),
                           "--output-last-message", str(response),
                           "-c", 'web_search="disabled"', "-c", "mcp_servers={}",
                           "-c", "model_instructions_file=" + json.dumps(str(instructions)),
                           "-c", "project_doc_max_bytes=0", "-c", 'developer_instructions=""',
                           "-c", "notify=[]"]
                for feature in _DISABLED_FEATURES:
                    command.extend(["--disable", feature])
                command.append("-")
                self.calls += 1
                usage.update(calls=1, call_number=self.calls, command=command)
                returncode, stdout, stderr, timed_out = _run(command, prompt, self.timeout_seconds, directory)
                events, other_lines = _events(stdout)
                usage.update(returncode=returncode, stderr=stderr, protocol_lines=other_lines)
                tool_events = []
                for event in events:
                    if event.get("model"):
                        usage["model"] = event["model"]
                    if event.get("type") == "turn.completed":
                        reported = event.get("usage", {})
                        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                            usage[key] = reported.get(key)
                    item = event.get("item", {})
                    if item.get("type") in {"command_execution", "mcp_tool_call", "web_search", "tool_call"}:
                        tool_events.append(item)
                    if event.get("type") in {"error", "turn.failed"}:
                        usage["errors"].append(event.get("message") or event.get("error") or event)
                usage["tool_events"] = tool_events
                if isinstance(usage["input_tokens"], int) and isinstance(usage["output_tokens"], int):
                    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
                    self.total_tokens += usage["total_tokens"]
                else:
                    self.usage_unavailable = True
                usage["cumulative_reported_tokens"] = self.total_tokens
                usage["token_budget_exceeded"] = self.total_tokens > self.token_budget
                if timed_out:
                    usage["status"] = "timeout"
                elif returncode:
                    usage["status"] = "runtime_failed"
                elif tool_events:
                    usage["status"] = "unexpected_tool_use"
                elif not response.exists():
                    usage["status"] = "missing_response"
                else:
                    raw_response = response.read_text(encoding="utf-8")
                    usage["response"] = raw_response
                    payload = json.loads(raw_response)
                    candidates, probes, errors = _parse_proposals(
                        payload, observations, spec, budget, call_number=self.calls,
                        max_candidates=self.max_candidates, max_probes=self.max_probes,
                        max_history=self.max_history)
                    usage["errors"].extend(errors)
                    usage["status"] = "completed" if candidates or probes or not errors else "invalid_proposals"
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
            usage.update(status="runtime_failed")
            usage["errors"].append(str(error))
        usage["wall_time_seconds"] = monotonic() - started
        usage["compute_cost"] = budget.spent - compute_before
        return ProposalBatch(candidates, probes, usage)
