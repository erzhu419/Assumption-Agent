"""Run frozen 0.21 cognition on the preselected fresh seed 6781."""

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts/v022_baseline_source_20260909/hegel_machine"
SUMMARY = ROOT / "artifacts/hegel_machine_v022_baseline_seed6781_summary_20260909.json"
spec = importlib.util.spec_from_file_location("hegel_machine", SNAPSHOT / "__init__.py",
    submodule_search_locations=[str(SNAPSHOT), str(ROOT / "src/hegel_machine")])
package = importlib.util.module_from_spec(spec)
sys.modules["hegel_machine"] = package
spec.loader.exec_module(package)
sys.path.insert(0, str(ROOT / "scripts"))
from run_method_session_study import run_study


def snapshot_modules():
    modules = [module.__file__ for name, module in sys.modules.items()
               if name.startswith("hegel_machine.cognition") and getattr(module, "__file__", None)]
    assert modules and all(str(SNAPSHOT / "cognition") in path for path in modules)
    return len(modules)


print({"baseline_cognition_modules": snapshot_modules(), "source": str(SNAPSHOT),
       "seed": 6781}, flush=True)
report = run_study(seed=6781, label="v022_baseline", date="20260909")
compact = {"seed": 6781, "source": str(SNAPSHOT), "baseline_version": package.__version__,
           "cognition_modules_from_snapshot": snapshot_modules(),
           "protocol": {"tasks": 65, "partitions": report["partitions"],
                        "limits_per_arm": report["limits_per_arm"], "evaluation": report["evaluation"]},
           "summary": {key: value for key, value in report["summary"].items() if key != "tasks"},
           "checkpoints": report["checkpoints"], "evaluations": report["evaluations"],
           "wall_time_seconds": report["wall_time_seconds"]}
with SUMMARY.open("x") as stream:
    json.dump(compact, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print({"baseline_complete": True, "summary": str(SUMMARY),
       "baseline_cognition_modules": snapshot_modules()}, flush=True)
