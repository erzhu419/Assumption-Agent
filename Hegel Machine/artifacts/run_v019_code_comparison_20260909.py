"""Fresh Hegel 0.19 against the retained September 8 ordinary Python runs."""

from dataclasses import asdict
import json
from pathlib import Path

from hegel_machine import __version__
from hegel_machine.cognition.code_agent import CodeAgentConfig
from hegel_machine.cognition.code_comparison import run_code_comparison
from hegel_machine.cognition.models import ResearchConfig


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/code_comparison_20260908.json"
OUTPUT = ROOT / "artifacts/code_comparison_v019_20260909.json"
SUMMARY = ROOT / "artifacts/code_comparison_v019_20260909_summary.json"


def no_model_provider(_):
    raise AssertionError("All ordinary Python results are retained; no new model call is authorized")


def main():
    original = json.loads(SOURCE.read_text())
    config = ResearchConfig(**original["config"])
    code_config = CodeAgentConfig(**original["code_config"])
    tasks = tuple(item["task_index"] for item in original["results"])
    assert tasks == (0, 1, 2)
    assert config.seed == 313 and original["backend"] == "process"
    assert (config.observation_budget, config.compute_budget, code_config.measurement_budget) == (128, 100000, 512)
    for item in original["results"]:
        assert item["seed"] == config.seed + 10007 * item["task_index"]
        assert item["code_agent"]["config"] == asdict(code_config)
        assert item["code_agent"]["task_index"] == item["task_index"]
        assert item["code_agent"]["status"] == "completed"
        assert item["code_agent"]["evaluation"]["rows"] == 256
    provenance = {
        "ordinary_code_source": str(SOURCE), "ordinary_code_run_date": "2026-09-08",
        "hegel_run_date": "2026-09-09", "hegel_version": __version__,
        "ordinary_code_results_reused": True, "new_model_calls": 0,
        "task_seeds": [item["seed"] for item in original["results"]],
        "comparison": "Retained ordinary Python runs versus fresh Hegel measurement research with matching task seeds and resource caps.",
        "computation": "Hegel search candidate-rows and Python scored candidate-rows are reported separately from Python execution and LLM tokens.",
    }

    def save(report):
        report["provenance"] = provenance
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        item = report["results"][-1]
        print(json.dumps({"task_index": item["task_index"], "seed": item["seed"],
            "hegel_mse": item["hegel"]["evaluation"]["mse"],
            "hegel_costs": item["hegel"]["costs"]}), flush=True)

    result = run_code_comparison(ROOT / "artifacts/code_comparison_v019_20260909_workspace",
        tasks=tasks, config=config, code_config=code_config,
        measurement_budget=code_config.measurement_budget, backend=original["backend"],
        provider_factory=no_model_provider,
        code_results=[item["code_agent"] for item in original["results"]],
        on_result=save)
    result["provenance"] = provenance
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    compact = {"provenance": provenance, "resource_contract": result["resource_contract"],
        "results": [{"task_index": item["task_index"], "seed": item["seed"],
            **{arm: {"channels": item[arm]["selected_channels"],
                "mse": item[arm]["evaluation"]["mse"], "evaluation_rows": item[arm]["evaluation"]["rows"],
                "evaluation_compute_cost": item[arm]["evaluation"]["compute_cost"],
                "costs": item[arm]["costs"],
                **({"token_budget_exceeded": item[arm]["token_budget_exceeded"]} if arm == "code_agent" else {})}
                for arm in ("code_agent", "hegel")}}
            for item in result["results"]]}
    SUMMARY.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "summary": str(SUMMARY), "new_model_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
