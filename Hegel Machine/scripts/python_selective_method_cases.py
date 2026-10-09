"""Real calibration-change cases and an offline comparison of two authored methods."""

from copy import deepcopy
import json
from pathlib import Path
from time import monotonic
import traceback

from scripts.run_python_method_learning_study import (
    InquiryLLMConfig, METHOD_PATH, case_spec as original_case_spec,
    execution_evidence, protocol as original_protocol, score_episode,
    selected_candidate, total_costs, trial_preparer, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
OLD_AUTHOR_RESULT = ROOT / "artifacts/python_method_learning_contract_luna_20261006/learning/result.json"
PUBLISHED_AUTHOR_RESULT = ROOT / "artifacts/python_method_publication_luna_20261006/learning/result.json"
SOURCE_OUTCOMES = OLD_AUTHOR_RESULT.parent.parent / "existing_source_outcomes.json"
KINDS = ("unchanged", "response_parameter_change", "recurrent_instrument_change", "post_validation_change")
POST_RESPONSE_OFFSET = 500009


def case_spec(source, kind, *, seed_offset, response_offset):
    """Keep existing physics; update only response parameters at the public checkpoint."""
    original_kind = "response_parameter_change" if kind == "post_validation_change" else kind
    spec = original_case_spec(source, original_kind,
        seed_offset=seed_offset, response_offset=response_offset)
    spec["kind"] = kind
    if kind == "post_validation_change":
        spec["task"]["calibration_change"] = {
            "after_rows": 96, "apparatus_seed": spec["task"]["apparatus_seed"],
            "response_seed": spec["task"]["response_seed"] + POST_RESPONSE_OFFSET}
    return spec


def protocol(sources):
    """Register four author cases and eight cross-source cases before authoring."""
    plan = original_protocol(sources)
    admitted = [(index, source) for index, source in enumerate(sources) if source["candidate"] is not None]
    author_index, author = admitted[0]
    plan.update(study="python_selective_method_development", target_kinds=list(KINDS),
        author_source_index=author_index,
        practice_cases={f"case_{index}": case_spec(author, kind,
            seed_offset=900011, response_offset=1000003) for index, kind in enumerate(KINDS)},
        transfer_cases=[{"id": f"target_{position}_{kind_index}", "source_index": source_index,
            **case_spec(source, kind, seed_offset=1100009, response_offset=1200007)}
            for position, (source_index, source) in enumerate(admitted[1:])
            for kind_index, kind in enumerate(KINDS)],
        calibration_change="The fourth case starts with changed response parameters and keeps "
            "the same finite-history instrument. At the first trajectory starting at or after "
            "96 paid rows the response parameters change again. The checkpoint and settled-state "
            "prediction objective are public; parameter values remain private.",
        intervention="Revise a previously authored research method on four anonymous practice "
            "cases, then freeze it for eight preregistered cross-source targets. Numerical "
            "operators, prefix and episode budgets are unchanged.",
        failure_policy="Preserve all author and practice attempts. If no method is published, "
            "skip all eight learned targets and retain the reference outcomes separately.",
        scope="Two existing archives times four conditions: unchanged, changed response, "
            "recurrent instrument, and a response change after the initial validation window. "
            "All are existing apparatus families; this is development evidence.")
    return plan


def _diagnostic_trace(method_results):
    """Index real actions and numerical feedback without inferring success from paid MSE."""
    trace = []
    for index, step in enumerate(method_results):
        proposal = step.get("proposal") or {}
        feedback = step.get("feedback") or {}
        item = {"step": index, "proposal": deepcopy(proposal)}
        for key in ("status", "error", "mse", "validation", "fresh_confirmation"):
            if key in feedback:
                item[key] = deepcopy(feedback[key])
        trace.append(item)
    return trace


def run_diagnostic(output):
    """Execute exact existing sources on the declared source-1 practice case_3."""
    output = Path(output).resolve()
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    plan = protocol(sources)
    source = sources[plan["author_source_index"]]
    spec = deepcopy(plan["practice_cases"]["case_3"])
    inputs = {"selective": OLD_AUTHOR_RESULT, "published": PUBLISHED_AUTHOR_RESULT}
    methods = {condition: json.loads(path.read_text())["outgoing_knowledge"]["files"][METHOD_PATH]
               for condition, path in inputs.items()}
    plan.update(study="python_post_validation_change_diagnostic", mode="offline_diagnostic",
        model=None, conditions=list(inputs), diagnostic_case=spec,
        source_results={condition: str(path) for condition, path in inputs.items()},
        scope="Two unchanged, previously authored methods on one existing source's public "
            "96-row calibration checkpoint. Both stop before terminal evaluation of the "
            "settled apparatus. This diagnoses a concrete branch, not new method learning.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    trained = []
    for condition, method in methods.items():
        (output / f"{condition}.py").write_text(method)
        path = output / condition
        payload = trial_preparer(source, plan)(spec, path, method, InquiryLLMConfig(**plan["limits"]))
        payload["agent"].confirmation_rng.seed(spec["confirmation_seed"])
        started = monotonic()
        try:
            payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                max_steps=plan["procedure_step_limit"])
        except Exception:
            payload["training"] = payload["agent"].result("execution_failed")
            payload["training"]["execution_error"] = traceback.format_exc()
        payload["research_wall_seconds"] = monotonic() - started
        write_json(path / "training.json", payload["training"])
        write_json(path / "paid_observations.json", payload["agent"].observations)
        trained.append((condition, payload))
    results = []
    for condition, payload in trained:
        result = score_episode(payload, plan)
        result.update(condition=condition, kind=spec["kind"], workspace=str(payload["path"]),
            unchanged_prior_selected=selected_candidate(payload) is not None,
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(payload["path"] / "evaluation.json", result)
        results.append(result)
    summary = {"study": plan["study"], "case": spec, "new_model_calls": 0,
        "new_reported_tokens": 0, "costs": total_costs([item["costs"] for item in results]),
        "conditions": {item["condition"]: {key: deepcopy(item[key]) for key in (
            "status", "confirmed_and_correct", "fresh_confirmed", "false_confirmation",
            "terminal", "costs", "operator_calls", "method_steps", "workspace")} for item in results},
        "traces": {item["condition"]: _diagnostic_trace(item["method_results"]) for item in results},
        "source_results": plan["source_results"], "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary
