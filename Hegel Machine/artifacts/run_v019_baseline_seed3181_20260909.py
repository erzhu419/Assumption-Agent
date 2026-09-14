"""Run the pre-change 0.18 cognition package on the specified new instance."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts/v019_baseline_source_20260909/hegel_machine"
spec = importlib.util.spec_from_file_location("hegel_machine", SNAPSHOT / "__init__.py",
    submodule_search_locations=[str(SNAPSHOT), str(ROOT / "src/hegel_machine")])
package = importlib.util.module_from_spec(spec)
sys.modules["hegel_machine"] = package
spec.loader.exec_module(package)
sys.path.insert(0, str(ROOT / "scripts"))
from run_method_session_study import run_study

run_study(seed=3181, label="v019_baseline", date="20260909")
modules = [module.__file__ for name, module in sys.modules.items()
           if name.startswith("hegel_machine.cognition") and getattr(module, "__file__", None)]
assert modules and all(str(SNAPSHOT / "cognition") in path for path in modules)
print({"baseline_cognition_modules": len(modules), "source": str(SNAPSHOT)}, flush=True)
