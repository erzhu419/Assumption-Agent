"""An ordinary Python coding agent with real experimental and execution tools."""

from dataclasses import asdict, dataclass
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from time import monotonic
from typing import Protocol

from .models import Observation, Probe
from .evaluation import point_loss
from .proposers import _events, _run


@dataclass(frozen=True)
class CodeAgentConfig:
    observation_budget: int = 128
    measurement_budget: int = 512
    compute_budget: int = 100000
    execution_seconds: float = 60.
    max_steps: int = 20
    max_model_calls: int = 16
    token_budget: int = 60000


class CodeProvider(Protocol):
    def next_action(self, request: dict) -> dict:
        """Return {action, arguments, usage}; usage includes reported tokens."""


_ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"action": {"type": "string", "enum": ["query", "write", "run", "evaluate", "finish"]},
                   "arguments_json": {"type": "string"}},
    "required": ["action", "arguments_json"]}

_INSTRUCTIONS = """You are a scientific coding agent. Infer the numerical process from paid observations.
Use the supplied query/write/run/evaluate/finish tools through one JSON action per turn.
You may write and execute unrestricted ordinary Python in your own working directory, fit parameters,
import available scientific libraries, inspect your own outputs, and repair programs after errors.
The environment is accessible through query only. Its implementation and final evaluation rows are not
part of this task. Do not inspect other directories or call external tools directly; return the action
for the supplied execution loop. Programs are ordinary Python, not an expression AST.
Model contract: model.py defines predict(row)->float; row contains features and history (past feature
values plus past __target__), never the current target. Optional fit(observations) runs on the acquired
training rows from the selected measurement view before predictions. Each query resets its trajectory. A fit script can also save local
parameters for model.py. Query controls must obey the public bounds. Channels are optional paid inputs.
When measurement_views are supplied, choose one listed channel combination. Each view lists its
measured features and public object roles; these can differ from the command names. For evaluate or
finish, channels selects the model's view. Compatible acquired rows are projected into that view;
coarse readings cannot recover individual readings. observations.json retains all purchased rows.
query, run, evaluate and finish refresh model_observations.json with compatible paid rows in the
selected view. Use this file for fitting and residuals; it excludes incompatible rows and projects
features and history consistently. A run action accepts channels; omitted channels use the latest
query view. The fit(observations) argument uses the same projected training rows.
Raw paid data are available in observations.json. The final model must work for new trajectories.
When a query also returns paid before/after measurements, its feedback names transitions.json.
This file retains batches with channels, observation_start (the zero-based offset in observations.json),
and transitions in the same order as that batch's observations. Iterate batches first, then each
batch["transitions"]; these are not a flat list of before/after records. Use these measured transitions to
study the process; they are part of the purchased evidence, not additional prediction inputs.
When session_knowledge is supplied, this is a continuing research programme. Reuse and revise its
ordinary Python libraries and previous lessons when the acquired evidence supports them. Files under
shared/ survive into later tasks; write reusable code and concise applicability/failure lessons there.
Other tasks can change their parameters or mechanism. Refit on the current observations instead of
blindly copying old coefficients. You choose what to preserve and how to use it. A finish action may
include lesson (text describing what was learned) and reused_files (shared paths actually used).
The request previews the latest 12 observations with their original row indices and trajectory_start
markers. observations.json contains every acquired row including its full history. research_context.json
contains the complete current request, including every earlier action, argument and tool output.
The request summarizes older actions and includes the latest tool feedback. Read the files with run
when you need more evidence, and print concise numerical summaries instead of whole datasets.
All source and current paid data remain available; the preview is not a fitting dataset.
Tools, with JSON arguments encoded in arguments_json:
query: {sequence:[{control:value},...],channels:[names]}; select probes and instrument channels yourself.
write: {path:'model.py',code:'ordinary Python source'}; also supports other local files.
run: {code:'ordinary Python to execute',channels:[names]} or {path:'fit.py',channels:[names]};
refreshes model_observations.json before execution and returns stdout/stderr and exit code.
evaluate: {model:'model.py',channels:[names]}; report training task loss in the selected view (channels optional).
finish: {model:'model.py',channels:[needed channel names]}; validates the executable then ends research.
The public risk_quantile fixes the loss: q=0.5 uses squared error; other q values use pinball loss
max(q*(target-prediction), (q-1)*(target-prediction)). Optimize the reported loss. MSE is also reported
as a squared-error diagnostic and can differ from the task loss. The risk target cannot be changed.
Prefer efficient programs and finish within the remaining calls. Use enough varied observations to
identify history dependence. All statistical fitting must use acquired observations, not guesses at
private coefficients. A bad execution is returned as feedback so you can repair it.
"""


class CodexCodeProvider:
    """Use the currently configured Codex model as the code-agent decision maker."""

    def __init__(self, workspace, *, executable=None, timeout_seconds=120, model=None, reasoning_effort=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.executable = executable or shutil.which("codex")
        self.timeout_seconds, self.model = timeout_seconds, model
        self.reasoning_effort = reasoning_effort
        self.calls = max([0, *(int(path.stem.rsplit("_", 1)[1])
                               for path in self.workspace.glob("code_action_[0-9][0-9][0-9].json"))])

    def next_action(self, request):
        if not self.executable:
            raise RuntimeError("Codex CLI was not found on PATH")
        self.calls += 1
        schema = self.workspace / "code_action_schema.json"
        instructions = self.workspace / "code_agent_instructions.txt"
        response = self.workspace / f"code_action_{self.calls:03d}.json"
        schema.write_text(json.dumps(_ACTION_SCHEMA), encoding="utf-8")
        instructions.write_text(_INSTRUCTIONS, encoding="utf-8")
        command = [self.executable, "-a", "never", "exec", "--sandbox", "read-only",
                   "--cd", str(self.workspace), "--skip-git-repo-check", "--ephemeral", "--json",
                   "--color", "never", "--output-schema", str(schema), "--output-last-message", str(response),
                   "-c", "model_instructions_file=" + json.dumps(str(instructions)),
                   "-c", "project_doc_max_bytes=0", "-c", 'developer_instructions=""',
                   "-c", 'web_search="disabled"', "-c", "mcp_servers={}", "-c", "notify=[]"]
        # The provider selects actions; the agent loop executes its ordinary
        # Python and apparatus calls and returns their actual outputs.
        for feature in ("shell_tool", "apps", "plugins", "memories", "multi_agent",
                        "browser_use", "computer_use", "image_generation",
                        "skill_search", "hooks", "view_image"):
            command.extend(("--disable", feature))
        # The installed CLI requires its host even when only returning a JSON action.
        command.extend(("--enable", "code_mode_host"))
        if self.model:
            command.extend(("--model", self.model))
        if self.reasoning_effort:
            command.extend(("-c", "model_reasoning_effort=" + json.dumps(self.reasoning_effort)))
        command.append("-")
        # Preserve the full research record for ordinary Python inspection;
        # repeated raw trajectories and old stdout exhausted real runs in 3 calls.
        (self.workspace / "research_context.json").write_text(json.dumps(request), encoding="utf-8")
        context = deepcopy(request)
        observations = context.get("observations", [])
        offset = max(0, len(observations) - 12)
        context["observations"] = [{**row, "row_index": offset + index}
                                   for index, row in enumerate(observations[offset:])]
        context["evidence"] = {"observations_file": "observations.json", "observation_count": len(observations),
                               "preview_start": offset, "full_context_file": "research_context.json"}
        events = context.get("events", [])
        for index, event in enumerate(events):
            arguments = event.get("arguments", {})
            compact_arguments = {key: value for key, value in arguments.items() if key not in ("sequence", "code")}
            if "sequence" in arguments:
                compact_arguments["sequence_rows"] = len(arguments["sequence"])
            if "code" in arguments:
                compact_arguments["path"] = arguments.get("path", f"scratch_{index:03d}.py")
            feedback = event.get("feedback", {})
            if index < len(events) - 1:
                feedback = {key: value for key, value in feedback.items() if key not in ("stdout", "stderr")}
            events[index] = {key: value for key, value in event.items() if key not in ("arguments", "feedback", "usage")}
            events[index].update(arguments=compact_arguments, feedback=feedback)
        knowledge = context.get("session_knowledge", {})
        if "files" in knowledge:
            knowledge["shared_files"] = list(knowledge["files"])
            knowledge["files"] = {name: code for name, code in knowledge["files"].items()
                                  if context.get("files", {}).get(name) != code}
        started = monotonic()
        code, stdout, stderr, timed_out = _run(command, json.dumps(context), self.timeout_seconds, str(self.workspace))
        (self.workspace / f"code_call_{self.calls:03d}.jsonl").write_text(stdout, encoding="utf-8")
        events, _ = _events(stdout)
        usage = {"backend": "codex_cli", "runtime": self.executable, "calls": 1,
                 "input_tokens": None, "output_tokens": None, "total_tokens": None,
                 "wall_time_seconds": monotonic() - started, "model": self.model or "cli_default"}
        if self.reasoning_effort:
            usage["requested_reasoning_effort"] = self.reasoning_effort
        errors = []
        for event in events:
            if event.get("model"):
                usage["model"] = event["model"]
            if event.get("type") == "turn.completed":
                usage.update({name: event.get("usage", {}).get(name)
                              for name in ("input_tokens", "output_tokens", "cached_input_tokens")})
            if event.get("type") in {"error", "turn.failed"}:
                errors.append(event.get("message") or event.get("error"))
            item = event.get("item", {})
            if item.get("type") == "error" and event.get("type") == "item.completed":
                errors.append(item.get("message"))
        if errors:
            usage["runtime_errors"] = errors
        if isinstance(usage["input_tokens"], int) and isinstance(usage["output_tokens"], int):
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if timed_out or code or not response.exists():
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"returncode": code, "timeout": timed_out, "messages": errors,
                              "stderr": stderr[-4000:]}}
        try:
            payload = json.loads(response.read_text(encoding="utf-8"))
            # Actual CLI completions sometimes retain literal line breaks in
            # this nested string. Preserve the authored code for execution.
            arguments = json.loads(payload["arguments_json"], strict=False)
        except (ValueError, KeyError, TypeError) as error:
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"decode_error": f"{type(error).__name__}: {error}"}}
        return {"action": payload["action"], "arguments": arguments, "usage": usage}


_MODEL_RUNNER = """import contextlib, json, runpy, sys
payload = json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    model = runpy.run_path(sys.argv[1])
    if callable(model.get('fit')):
        model['fit'](json.load(open(sys.argv[2])))
    predictions = [float(model['predict'](row)) for row in payload]
print(json.dumps(predictions, allow_nan=False))
"""


def evaluate_python_model(workspace, model, observations, *, timeout_seconds=30, risk_quantile=.5):
    """Run a Python model on input rows; current evaluation targets stay outside."""
    workspace = Path(workspace).resolve()
    payload = [{"features": row.features, "history": row.history} if hasattr(row, "features")
               else {"features": row["features"], "history": row.get("history", [])} for row in observations]
    # A retained candidate carries its selected paid training view to terminal
    # execution. Terminal prediction rows never replace these fitting rows.
    training = workspace / "model_observations.json"
    if not training.exists():
        training = workspace / "observations.json"
    code, stdout, stderr, timed_out = _run([sys.executable, "-c", _MODEL_RUNNER, str(workspace / model), str(training)],
                                         json.dumps(payload), timeout_seconds, str(workspace))
    result = {"returncode": code, "stderr": stderr, "timeout": timed_out,
              "risk_quantile": risk_quantile, "loss_kind": "mse" if risk_quantile == .5 else "pinball"}
    if code == 0:
        predictions = json.loads(stdout)
        if len(predictions) != len(observations):
            raise ValueError("model returned a different number of predictions")
        targets = [row.target if hasattr(row, "target") else row["target"] for row in observations]
        result.update(predictions=predictions, mse=sum((value - target) ** 2
                      for value, target in zip(predictions, targets)) / max(1, len(targets)))
        result["loss"] = sum(point_loss(value, target, risk_quantile)
                             for value, target in zip(predictions, targets)) / max(1, len(targets))
    return result


def retain_evaluated_python_model(workspace, model, channels, training_mse, *,
                                  training_loss=None, loss_kind="mse", risk_quantile=.5):
    """Keep a successfully executed candidate while ordinary work continues."""
    workspace = Path(workspace).resolve()
    retained = workspace / "_evaluated_model"
    if retained.exists():
        shutil.rmtree(retained)
    retained.mkdir()
    for path in workspace.iterdir():
        if (path.name in {"_evaluated_model", "__pycache__", "code_agent_instructions.txt"}
                or path.name.startswith(("code_action_", "code_call_"))):
            continue
        target = retained / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    return {"workspace": str(retained), "model": model, "selected_channels": list(channels),
            "training_mse": training_mse, "model_source": (retained / model).read_text(),
            "training_loss": training_mse if training_loss is None else training_loss,
            "loss_kind": loss_kind, "risk_quantile": risk_quantile}


class CodeAgent:
    """Execute the model's experimental and Python decisions with real feedback."""

    def __init__(self, provider, workspace, config=None, *, state=None, session_knowledge=None):
        self.provider, self.workspace = provider, Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config or CodeAgentConfig()
        self.observations, self.events, self.files = [], [], {}
        self.transition_batches = []
        self.costs = {"observation_cost": 0, "measurement_cost": 0, "compute_cost": 0,
                      "python_execution_seconds": 0., "model_calls": 0, "reported_tokens": 0}
        self.tokens_available = True
        self.model = None
        self.selected_channels = []
        self.session_knowledge = deepcopy(session_knowledge or {})
        self.lesson, self.reused_files = "", []
        self.evaluated_model = None
        if state is not None:
            self.observations = [Observation(dict(row["features"]), row["target"],
                tuple(dict(item) for item in row.get("history", []))) for row in state["observations"]]
            self.events, self.files, self.costs = deepcopy(state["events"]), deepcopy(state["files"]), dict(state["costs"])
            self.tokens_available = state["tokens_available"]
            self.model, self.selected_channels = state["model"], list(state["selected_channels"])
            self.lesson, self.reused_files = state.get("lesson", ""), state.get("reused_files", [])
            self.evaluated_model = deepcopy(state.get("evaluated_model"))
            self.transition_batches = deepcopy(state.get("transition_batches", []))
            for name, source in self.files.items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            (self.workspace / "observations.json").write_text(json.dumps([asdict(row) for row in self.observations]))
            if self.transition_batches:
                (self.workspace / "transitions.json").write_text(json.dumps(self.transition_batches), encoding="utf-8")
        else:
            for name, source in self.session_knowledge.get("files", {}).items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
                self.files[name] = source

    def to_dict(self):
        names = set(self.files)
        names.update(str(path.relative_to(self.workspace)) for path in (self.workspace / "shared").rglob("*") if path.is_file())
        if self.model:
            names.add(self.model)
        for name in names:
            path = self.workspace / name
            if path.is_file():
                try:
                    self.files[name] = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass  # Binary fit artifacts stay in this persistent task workspace.
        return {"observations": [asdict(row) for row in self.observations],
                "transition_batches": deepcopy(self.transition_batches),
                "events": deepcopy(self.events), "files": dict(self.files), "costs": dict(self.costs),
                "tokens_available": self.tokens_available, "model": self.model,
                "selected_channels": list(self.selected_channels), "lesson": self.lesson,
                "reused_files": list(self.reused_files), "evaluated_model": deepcopy(self.evaluated_model)}

    def _execute(self, action, arguments, laboratory):
        if action == "query":
            probe = Probe(f"code_query_{len(self.events)}", tuple(arguments["sequence"]))
            channels = tuple(arguments.get("channels", ()))
            rates = {channel.name: channel.cost_per_row for channel in getattr(laboratory, "channels", ())}
            units = probe.cost * sum(rates[name] for name in set(channels))
            if self.costs["observation_cost"] + probe.cost > self.config.observation_budget:
                return {"error": "observation budget exhausted"}
            if self.costs["measurement_cost"] + units > self.config.measurement_budget:
                return {"error": "measurement budget exhausted"}
            transitions = []
            if hasattr(laboratory, "acquire"):
                batch = laboratory.acquire(probe, channels)
                rows, units = batch.observations, batch.measurement_cost
                transitions = getattr(batch, "transitions", [])
            else:
                rows = laboratory.observe(probe)
            if transitions:
                self.transition_batches.append({"channels": list(channels),
                    "observation_start": len(self.observations), "transitions": deepcopy(transitions)})
                (self.workspace / "transitions.json").write_text(json.dumps(self.transition_batches), encoding="utf-8")
            self.observations.extend(rows)
            self.selected_channels = list(channels)
            self.costs["observation_cost"] += len(rows)
            self.costs["measurement_cost"] += units
            (self.workspace / "observations.json").write_text(json.dumps([asdict(row) for row in self.observations]), encoding="utf-8")
            self._write_model_observations(laboratory, channels)
            feedback = {"observations_file": "observations.json", "model_observations_file": "model_observations.json",
                        "observation_cost": len(rows), "measurement_cost": units}
            if transitions:
                feedback.update(transitions_file="transitions.json", transition_count=len(transitions),
                    transitions_layout="batches: channels, observation_start, transitions[{before, after}]")
            if hasattr(laboratory, "measurement_views"):
                feedback.update(channels=list(channels),
                    feature_names=list(laboratory.spec_for(channels).feature_names))
            return feedback
        if action == "write":
            path = self.workspace / arguments["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["code"], encoding="utf-8")
            self.files[arguments["path"]] = arguments["code"]
            return {"written": arguments["path"]}
        remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining <= 0:
            return {"error": "Python execution time budget exhausted"}
        if action == "run":
            channels = tuple(arguments.get("channels", self.selected_channels))
            rows = self._write_model_observations(laboratory, channels)
            path = arguments.get("path", f"scratch_{len(self.events):03d}.py")
            if "code" in arguments:
                (self.workspace / path).write_text(arguments["code"], encoding="utf-8")
                self.files[path] = arguments["code"]
            started = monotonic()
            code, stdout, stderr, timed_out = _run([sys.executable, str(self.workspace / path)], "", remaining, str(self.workspace))
            self.costs["python_execution_seconds"] += monotonic() - started
            return {"returncode": code, "stdout": stdout[-12000:], "stderr": stderr[-12000:], "timeout": timed_out,
                    "model_observations_file": "model_observations.json", "model_observation_count": len(rows),
                    "channels": list(channels)}
        if action in {"evaluate", "finish"}:
            channels = tuple(arguments.get("channels", self.selected_channels))
            observations = self._write_model_observations(laboratory, channels)
            if not observations:
                return {"error": "acquire observations before fitting a model"}
            if self.costs["compute_cost"] + len(observations) > self.config.compute_budget:
                return {"error": "candidate-row scoring budget exhausted"}
            model = arguments.get("model", "model.py")
            self.costs["compute_cost"] += len(observations)
            started = monotonic()
            result = evaluate_python_model(self.workspace, model, observations,
                timeout_seconds=remaining, risk_quantile=laboratory.spec.risk_quantile)
            self.costs["python_execution_seconds"] += monotonic() - started
            result.pop("predictions", None)
            if action == "evaluate" and result["returncode"] == 0:
                self.evaluated_model = retain_evaluated_python_model(self.workspace, model,
                    arguments.get("channels", self.selected_channels), result["mse"],
                    training_loss=result["loss"], loss_kind=result["loss_kind"],
                    risk_quantile=result["risk_quantile"])
            if action == "finish" and result["returncode"] == 0:
                self.model = model
                self.selected_channels = list(arguments.get("channels", self.selected_channels))
                self.lesson = arguments.get("lesson", "")
                self.reused_files = list(arguments.get("reused_files", []))
            return result
        return {"error": f"unknown tool: {action}"}

    def _write_model_observations(self, laboratory, channels):
        observations = self._model_observations(laboratory, channels)
        (self.workspace / "model_observations.json").write_text(
            json.dumps([asdict(row) for row in observations]), encoding="utf-8")
        return observations

    def _model_observations(self, laboratory, channels):
        if not hasattr(laboratory, "measurement_views"):
            if not hasattr(laboratory, "spec_for"):
                return self.observations
            names = set(laboratory.spec_for(channels).feature_names)
            return [Observation({name: row.features[name] for name in names}, row.target,
                        tuple({name: value for name, value in point.items()
                               if name in names or name == "__target__"} for point in row.history))
                    for row in self.observations if names <= set(row.features)]
        laboratory.spec_for(channels)
        views = [(view, set(laboratory.spec_for(view).feature_names))
                 for view in laboratory.measurement_views]
        observations = []
        for row in self.observations:
            source = next(view for view, names in views if set(row.features) == names)
            if channels in laboratory.projectable_views(source):
                observations.extend(laboratory.project_rows([row], source, channels))
        return observations

    def run(self, laboratory, *, on_checkpoint=None):
        spec = laboratory.spec
        capabilities = {"description": spec.description, "feature_names": spec.feature_names,
                        "feature_bounds": spec.feature_bounds, "max_probe_length": spec.max_probe_length,
                        "risk_quantile": spec.risk_quantile,
                        "loss_kind": "mse" if spec.risk_quantile == .5 else "pinball",
                        "channels": [asdict(channel) for channel in getattr(laboratory, "channels", ())],
                        "python_libraries": [name for name in ("numpy", "scipy") if importlib.util.find_spec(name)]}
        if hasattr(laboratory, "measurement_views"):
            capabilities["measurement_views"] = [{"channels": list(view),
                "feature_names": list(laboratory.spec_for(view).feature_names),
                "context": dict(laboratory.spec_for(view).context),
                "projectable_to": [list(target) for target in laboratory.projectable_views(view)]}
                for view in laboratory.measurement_views]
        status = "completed" if self.model else "step_budget_exhausted"
        for step in range(len(self.events), self.config.max_steps):
            if self.model:
                break
            if self.costs["model_calls"] >= self.config.max_model_calls:
                status = "model_call_budget_exhausted"
                break
            if not self.tokens_available or self.costs["reported_tokens"] >= self.config.token_budget:
                status = "token_usage_unavailable" if not self.tokens_available else "token_budget_exhausted"
                break
            request = {"capabilities": capabilities, "limits": asdict(self.config), "costs": dict(self.costs),
                       "observations": [{"features": row.features, "target": row.target,
                                          "trajectory_start": not row.history} for row in self.observations],
                       "files": dict(self.files),
                       "session_knowledge": self.session_knowledge,
                       "events": self.events}
            self.costs["model_calls"] += 1
            try:
                decision = self.provider.next_action(request)
            except Exception as error:
                self.tokens_available = False
                self.events.append({"action": "provider_error", "error": f"{type(error).__name__}: {error}"})
                status = "provider_failed"
                break
            usage = decision.get("usage", {})
            tokens = usage.get("total_tokens")
            if isinstance(tokens, int):
                self.costs["reported_tokens"] += tokens
            else:
                self.tokens_available = False
            action = decision["action"]
            if action == "provider_error":
                if decision.get("error", {}).get("decode_error"):
                    # A completed model reply with malformed tool arguments
                    # remains a paid turn. Let the model repair its own JSON
                    # within the same task's remaining call/token budget.
                    self.events.append({"step": step, "action": action, "arguments": {},
                        "feedback": {"error": decision["error"]["decode_error"]}, "usage": usage})
                    if on_checkpoint:
                        on_checkpoint(self.to_dict())
                    continue
                self.events.append(decision)
                status = "provider_failed"
                break
            try:
                feedback = self._execute(action, decision.get("arguments", {}), laboratory)
            except Exception as error:
                feedback = {"error": f"{type(error).__name__}: {error}"}
            self.events.append({"step": step, "action": action, "arguments": decision.get("arguments", {}),
                                "feedback": feedback, "usage": usage})
            if on_checkpoint:
                on_checkpoint(self.to_dict())
            if self.model is not None:
                status = "completed"
                break
        usable = ({"model": self.model, "workspace": str(self.workspace),
                   "model_source": (self.workspace / self.model).read_text(), "selected_channels": self.selected_channels}
                  if self.model else self.evaluated_model or {})
        return {"agent": "ordinary_python_code_agent", "status": status, "model": usable.get("model"),
                "risk_quantile": spec.risk_quantile,
                "loss_kind": "mse" if spec.risk_quantile == .5 else "pinball",
                "model_source": usable.get("model_source"),
                "model_origin": "finish" if self.model else "last_successful_evaluation" if self.evaluated_model else None,
                "selected_channels": usable.get("selected_channels", self.selected_channels),
                "workspace": usable.get("workspace", str(self.workspace)), "research_workspace": str(self.workspace),
                "lesson": self.lesson, "reused_files": self.reused_files,
                "costs": {**self.costs, "reported_tokens": self.costs["reported_tokens"] if self.tokens_available else None},
                "token_limit_kind": "reported_usage_between_calls", "config": asdict(self.config),
                "token_budget_exceeded": (self.costs["reported_tokens"] > self.config.token_budget
                                           if self.tokens_available else None),
                "events": self.events}
