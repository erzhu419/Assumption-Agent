#!/usr/bin/env python3
"""Same-LLM code / prose / executable-memory comparison.

Example (direct paid API):
 PYTHONPATH=src python scripts/run_online_inquiry_llm.py --model qwen3.8-flash \
   --backend chat_completions --profile development --cases 1 --output artifacts/inquiry_llm_dev

Default connection: RUOLI_GPT_KEY and RUOLI_BASE_URL in the process environment.
Use --base-url and --env-key for another explicitly selected provider connection.
The codex_cli backend uses the local Codex CLI authenticated with ChatGPT.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.online_inquiry_llm import (
    ARMS, InquiryLLMAgent, InquiryLLMConfig, evaluate_chain, run_chain)
from hegel_machine.cognition.responses_code_provider import ResponsesCodeProvider
from hegel_machine.cognition.chat_completions_code_provider import ChatCompletionsCodeProvider
from hegel_machine.cognition.code_agent import CodexCodeProvider
from scripts.run_online_inquiry_study import cohort_protocol, make_lab


def load_provider_environment(path, *, base_url=None, env_key="RUOLI_GPT_KEY"):
    """Load the selected key and default base URL; never serialize secrets."""
    if path:
        for line in Path(path).read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key in {"RUOLI_BASE_URL", env_key}:
                os.environ[key] = value.strip().strip("\"'")
    if not os.environ.get(env_key):
        raise ValueError(f"{env_key} is required; no provider fallback")
    base = (base_url if base_url is not None else os.environ.get("RUOLI_BASE_URL", "")).rstrip("/")
    if not base:
        raise ValueError("a base URL is required: use --base-url or RUOLI_BASE_URL")
    if not urlsplit(base).path:
        base += "/v1"
    return base


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_codex_subscription():
    status = subprocess.run(["codex", "login", "status"], capture_output=True, text=True, timeout=30)
    if status.returncode or "Logged in using ChatGPT" not in status.stdout + status.stderr:
        raise ValueError("codex_cli requires a locally verified ChatGPT login")
    return "ChatGPT_subscription"


def load_gemini_cache(path, *, model, backend):
    if backend != "chat_completions":
        raise ValueError("--gemini-cache-file requires --backend chat_completions")
    cache = json.loads(Path(path).read_text())
    if cache["model"].removeprefix("models/") != model.removeprefix("models/"):
        raise ValueError("Gemini cache model differs from the selected model")
    metadata = {key: cache[key] for key in ("name", "model", "task_interface")}
    metadata.update({key: cache[key] for key in ("expireTime", "usageMetadata") if key in cache})
    return metadata


def protocol(*, model, profile="development", cases=1, full_transfer=False,
             config=None, base_url=None, max_output_tokens=4096, timeout_seconds=120,
             backend="responses", env_key="RUOLI_GPT_KEY", gemini_cache=None,
             reasoning_effort=None, auth_mode=None):
    if not model:
        raise ValueError("the user must choose an explicit external model")
    if backend not in {"responses", "chat_completions", "codex_cli"}:
        raise ValueError("unsupported external provider backend")
    if gemini_cache and backend != "chat_completions":
        raise ValueError("Gemini cache requires the chat_completions backend")
    shared = cohort_protocol(profile)
    if not 1 <= cases <= len(shared["cases"]):
        raise ValueError("case count exceeds the registered cohort")
    chain_specs = []
    for index, case in enumerate(shared["cases"][:cases]):
        target_index = index % 4 if profile == "main" else 1
        consumers = case["consumers"] if full_transfer else [case["consumers"][target_index]]
        chain_specs.append({"chain_id": case["chain_id"], "source": case["source"],
                            "consumers": consumers})
    return {"study": "same_llm_online_inquiry_knowledge_conditions", "profile": profile,
            "model": model, "base_url": None if backend == "codex_cli" else base_url,
            "arms": list(ARMS), "cases": chain_specs,
            "limits": asdict(config or InquiryLLMConfig()), "evaluation": shared["evaluation"],
            "provider": {"backend": backend, "env_key": None if backend == "codex_cli" else env_key,
                         "max_output_tokens": None if backend == "codex_cli" else max_output_tokens,
                         "timeout_seconds": timeout_seconds,
                         **({"runtime": "codex_cli", "auth_mode": auth_mode,
                             "reasoning_effort": reasoning_effort} if backend == "codex_cli" else {}),
                         **({"gemini_cache": gemini_cache} if gemini_cache else {})},
            "memory_conditions": {"code_react": "task-local code and active experimentation",
                "textual_memory": "source agent-authored textual lesson only",
                "hegel": "source agent-authored executable files and editable question/hypothesis manifest"},
            "source_freeze": "Each arm pays to learn its own source; consumers independently inherit only that source.",
            "target_allocation": "Main single-target chains rotate the four fixed transfer cells by cohort position.",
            "terminal_order": "All registered arms and chains stop training before any terminal evaluation.",
            "failure_policy": "Keep initial provider, parsing, execution, budget and research failures; no retry replacement.",
            "format_error_policy": "Malformed model actions with reported usage receive charged corrective feedback within the original limits; no invalid action is executed. Service failures remain terminal.",
            "service_stop_policy": "Stop new chains after three consecutive chains end with model_unavailable or empty_response; finish in-flight chains and keep all records. No retry.",
            "scope": "Same-LLM knowledge representation and persistence intervention, not the offline solver's score.",
            "token_limit_kind": "reported usage between calls; the final paid call can exceed the cap",
            "python_limit_kind": "actual subprocess wall time, identical per-task cap; no host fitting solver"}


def tasks_for(case):
    return [{**item["lab"], "evaluation_seed": item["evaluation_seed"],
             "task_id": item["task_id"]} for item in [case["source"], *case["consumers"]]]


def laboratory(task):
    return make_lab({key: value for key, value in task.items()
                     if key not in {"evaluation_seed", "task_id", "confirmation_seed"}})


def service_failure_reason(chain):
    last = chain["tasks"][-1]
    if last["status"] != "provider_failed":
        return None
    error = last["events"][-1].get("error", {}) if last["events"] else {}
    if error.get("kind") == "empty_response":
        return "empty_response"
    if "model_unavailable" in json.dumps(error):
        return "model_unavailable"
    return None


def run_study(plan, output, *, provider_factory, lab_factory=laboratory, workers=1,
              agent_class=InquiryLLMAgent):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory; preserve the earlier run")
    write_json(output / "protocol.json", plan)
    results = []
    config = InquiryLLMConfig(**plan["limits"])
    def train(case, arm):
        workspace = output / case["chain_id"] / arm
        workspace.mkdir(parents=True, exist_ok=True)
        result = run_chain(tasks_for(case), arm=arm, workspace=workspace,
            provider_factory=provider_factory, lab_factory=lab_factory, config=config,
            agent_class=agent_class,
            on_checkpoint=lambda data, path=workspace: write_json(path / "active.json", data))
        result["chain_id"] = case["chain_id"]
        write_json(workspace / "training.json", result)
        print(json.dumps({"chain": case["chain_id"], "arm": arm,
                          "training_statuses": [task["status"] for task in result["tasks"]]}), flush=True)
        return result
    arms = tuple(plan["arms"])
    jobs = [(case, arm) for index, case in enumerate(plan["cases"])
            for arm in arms[index % len(arms):] + arms[:index % len(arms)]]
    stop_reason = None
    job_iterator = iter(jobs)
    unavailable = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(workers):
            job = next(job_iterator, None)
            if job is not None:
                pending.add(executor.submit(train, *job))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                results.append(result)
                reason = service_failure_reason(result)
                unavailable = unavailable + 1 if reason else 0
                if unavailable >= 3:
                    stop_reason = stop_reason or reason
            if stop_reason is None:
                for _ in finished:
                    job = next(job_iterator, None)
                    if job is not None:
                        pending.add(executor.submit(train, *job))
    order = {(case["chain_id"], arm): index for index, (case, arm) in enumerate(jobs)}
    results.sort(key=lambda result: order[(result["chain_id"], result["arm"])])
    for result in results:
        case = next(case for case in plan["cases"] if case["chain_id"] == result["chain_id"])
        evaluate_chain(result, tasks_for(case), lab_factory=lab_factory,
                       length=plan["evaluation"]["length"], timeout_seconds=config.execution_seconds)
        write_json(output / result["chain_id"] / result["arm"] / "result.json", result)
    summary = summarize(plan, results)
    if stop_reason:
        summary["status"] = "interrupted_service_failure"
        summary["stop_reason"] = stop_reason
    write_json(output / "summary.json", summary)
    return summary


def summarize(plan, results):
    summary = {"study": plan["study"], "model": plan["model"], "arms": {},
               "status": "completed" if len(results) == len(plan["cases"]) * len(plan["arms"])
                         else "interrupted_provider_unavailable",
               "evidence_kind": "real_external_model" if all(
                   event.get("usage", {}).get("backend") == plan["provider"]["backend"]
                   for chain in results for task in chain["tasks"] for event in task["events"])
                   else "interface_test_not_llm_performance"}
    for arm in plan["arms"]:
        chains = [chain for chain in results if chain["arm"] == arm]
        tasks = [task for chain in chains for task in chain["tasks"]]
        scores = [score for chain in chains for score in chain["evaluation"]]
        events = [event for task in tasks for event in task["events"]]
        usage = [event.get("usage", {}) for event in events]
        costs = {name: None if any(task["costs"][name] is None for task in tasks)
                 else sum(task["costs"][name] for task in tasks)
                 for name in ("observation_cost", "compute_cost", "python_execution_seconds",
                              "model_calls", "reported_tokens")}
        correct = [score.get("mse") is not None and score["mse"] <= plan["evaluation"]["mse_threshold"]
                   for score in scores]
        confirmed = [bool(task.get("fresh_confirmation", {}).get("passed"))
                     if task.get("fresh_confirmation") else False for task in tasks]
        summary["arms"][arm] = {"registered_tasks": sum(1 + len(case["consumers"]) for case in plan["cases"]),
            "attempted_tasks": len(tasks), "correct": sum(correct),
            "not_attempted_tasks": sum(1 + len(case["consumers"]) for case in plan["cases"]) - len(tasks),
            "infrastructure_failed_tasks": sum(task["status"] == "token_usage_unavailable" or (
                task["status"] == "provider_failed" and "decode_error" not in task["events"][-1].get("error", {}))
                for task in tasks),
            "provider_protocol_failed_tasks": sum(task["status"] == "provider_failed" and
                "decode_error" in task["events"][-1].get("error", {}) for task in tasks),
            "model_format_errors": sum("decode_error" in event.get("error", {}) for event in events),
            "completed_tasks": sum(task["status"] == "completed" for task in tasks),
            "incorrect_completed_tasks": sum(task["status"] == "completed" and not passed
                                              for task, passed in zip(tasks, correct)),
            "confirmed_and_correct": sum(a and b for a, b in zip(confirmed, correct)),
            "false_confirmation": sum(a and not b for a, b in zip(confirmed, correct)),
            "source_correct": sum(chain["evaluation"][0].get("mse", float("inf")) is not None and
                chain["evaluation"][0].get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"]
                for chain in chains),
            "costs_including_source": costs,
            "measurement_cost": sum(task["costs"].get("measurement_cost", 0) for task in tasks),
            "reported_tokens_known_lower_bound": sum(u["total_tokens"] for u in usage
                                                       if isinstance(u.get("total_tokens"), int)),
            "calls_without_token_usage": sum(not isinstance(u.get("total_tokens"), int) for u in usage),
            "cache_hit_calls": sum(isinstance(u.get("cached_input_tokens"), int)
                                   and u["cached_input_tokens"] > 0 for u in usage),
            "cached_input_tokens_known_lower_bound": sum(u["cached_input_tokens"] for u in usage
                                                          if isinstance(u.get("cached_input_tokens"), int)),
            "input_tokens_known_lower_bound": sum(u["input_tokens"] for u in usage
                                                    if isinstance(u.get("input_tokens"), int)),
            "calls_without_cache_usage": sum(not isinstance(u.get("cached_input_tokens"), int) for u in usage),
            "returned_models": sorted({u["returned_model"] for u in usage if u.get("returned_model")}),
            "inherited_entrypoint_attempts": sum(bool(e.get("feedback", {}).get("inherited_program")) for e in events),
            "inherited_entrypoint_successes": sum(bool(e.get("feedback", {}).get("inherited_program"))
                and not e.get("feedback", {}).get("error") and e.get("feedback", {}).get("returncode", 0) == 0
                for e in events),
            "unchanged_inherited_entrypoint_successes": sum(bool(e.get("feedback", {}).get("inherited_source_unchanged"))
                and not e.get("feedback", {}).get("error") and e.get("feedback", {}).get("returncode", 0) == 0
                for e in events),
            "inherited_file_revisions": sum(len(task.get("inherited_files_changed", [])) for task in tasks),
            "reuse_measurement": "Explicit query/run/evaluate/finish entrypoints; indirect Python imports are not traced.",
            "statuses": [task["status"] for task in tasks]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("responses", "chat_completions", "codex_cli"), default="responses")
    parser.add_argument("--reasoning-effort", help="Codex CLI model reasoning effort")
    parser.add_argument("--base-url", help="API base URL; defaults to RUOLI_BASE_URL")
    parser.add_argument("--env-key", default="RUOLI_GPT_KEY", help="environment variable containing the API key")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--gemini-cache-file", type=Path, help="metadata for an existing Gemini cached context")
    parser.add_argument("--profile", choices=("development", "main"), default="development")
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--full-transfer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-calls", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=60000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cache = (load_gemini_cache(args.gemini_cache_file, model=args.model, backend=args.backend)
             if args.gemini_cache_file else None)
    if args.backend == "codex_cli":
        base, env_key = None, None
        auth_mode = verify_codex_subscription()
    else:
        base = load_provider_environment(args.env_file, base_url=args.base_url, env_key=args.env_key)
        env_key, auth_mode = args.env_key, None
    config = InquiryLLMConfig(max_model_calls=args.max_model_calls, token_budget=args.token_budget)
    plan = protocol(model=args.model, profile=args.profile, cases=args.cases,
                    full_transfer=args.full_transfer, config=config, base_url=base,
                    max_output_tokens=args.max_output_tokens, timeout_seconds=args.timeout_seconds,
                    backend=args.backend, env_key=env_key, gemini_cache=cache,
                    reasoning_effort=args.reasoning_effort, auth_mode=auth_mode)
    provider_type = {"responses": ResponsesCodeProvider,
                     "chat_completions": ChatCompletionsCodeProvider,
                     "codex_cli": CodexCodeProvider}[args.backend]
    cache_options = ({"cached_content": cache["name"], "cached_task_interface": cache["task_interface"]}
                     if cache else {})
    def provider(path):
        if args.backend == "codex_cli":
            return provider_type(path, model=args.model, timeout_seconds=args.timeout_seconds,
                                 reasoning_effort=args.reasoning_effort)
        return provider_type(path, base_url=base, model=args.model, env_key=args.env_key,
            timeout_seconds=args.timeout_seconds, max_output_tokens=args.max_output_tokens, **cache_options)
    print(json.dumps(run_study(plan, args.output, provider_factory=provider, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
