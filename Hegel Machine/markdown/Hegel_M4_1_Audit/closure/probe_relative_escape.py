"""Small M4.1 model path containment control, confined to own scratch files."""
from pathlib import Path
import json
from hegel_machine.cognition.m4_compositional_discovery import evaluate_isolated_python_model

out = Path(__file__).resolve().parent
ws = out / 'current_semantics_workspaces/path_in'
ws.mkdir(parents=True, exist_ok=True)
external = out / 'current_semantics_workspaces/path_outside.py'
external.write_text('def predict(row):\n    return 42.0\n')
# TemporaryDirectory uses /tmp/hegel-m4-sandbox-*/ws in this environment.
relative = '../../../' + str(external).lstrip('/')
r = evaluate_isolated_python_model(
    ws, relative, [{'features': {'x': 1.}, 'target': 42.}], isolated_prefix_rows=[]
)
res = {
    'actual_commit': '99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c',
    'workspace': str(ws), 'out_of_workspace_model': str(external), 'relative_input_path': relative,
    'result': r,
    'scope': 'Only own scratch file is read. Demonstrates missing relative containment check, not independent data leakage.',
}
(out / 'relative_escape_result.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
