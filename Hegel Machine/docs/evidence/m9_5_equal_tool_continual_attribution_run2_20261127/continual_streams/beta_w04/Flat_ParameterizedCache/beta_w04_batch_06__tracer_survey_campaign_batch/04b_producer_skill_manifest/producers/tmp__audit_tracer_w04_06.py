import importlib.util
import json
import pickle
import tempfile
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load('beta_w04_batch_06_solver')
# Load the certificate verifier without running the validator's CLI.
import ast
from types import SimpleNamespace
path = root / 'validate_public_schema.py'
tree = ast.parse(path.read_text())
tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) or (isinstance(n, ast.FunctionDef) and n.name == 'verify_box_simplex_kkt_certificate')]
ns = {}
exec(compile(tree, str(path), 'exec'), ns)
v = SimpleNamespace(**ns)
with (root / 'public_beta_w04_batch_06.pkl').open('rb') as f:
    p = pickle.load(f)[0]
s, gt = p['study'], p['gt']
history = json.loads(m._HISTORY.read_text())
V = s['V_reaction_2d']
interaction = V - V.mean(0)[None, :] - V.mean(1)[:, None] + V.mean()
sep = np.linalg.norm(interaction) / np.linalg.norm(V)
assert sep <= 1e-6, sep
A, _ = m._forward(s)
Z, ridge = s['Z_nuisance'], s['nuisance_ridge']
S = m._profile(Z, A, ridge)
design_rel = np.linalg.norm(S - gt['S_tilde']) / np.linalg.norm(gt['S_tilde'])
assert design_rel < 1e-10, design_rel
wing = np.asarray(s['row_bank_id']) // 2
worst_wing = np.zeros(2)
worst_gap = 0.
worst_field = 0.
checked = []
with tempfile.TemporaryDirectory() as d:
    alternate = Path(d) / 'history.json'
    for i in range(len(history)):
        if len(set(history[i]['bank_id'])) != 4:
            continue
        alternate.write_text(json.dumps(history[:i] + history[i+1:]))
        m._HISTORY = alternate
        out = m.solve_beta_w04_batch_06(s)
        r = m._profile(Z, m._calibrate(s), ridge)
        worst_wing = np.maximum(worst_wing, [np.max(np.abs(r[wing == j] - gt['r_y'][wing == j])) for j in range(2)])
        check = v.verify_box_simplex_kkt_certificate(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
        assert check['valid'], (i, check)
        worst_gap = max(worst_gap, check['gap'])
        field = np.linalg.norm(out['terminal_field'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
        worst_field = max(worst_field, field)
        checked.append(i)
assert checked and max(worst_wing) < 1e-5 and worst_field < 1e-5, (checked, worst_wing, worst_field)
print('separability', sep, 'design_relative_error', design_rel)
print('held_out_complete_sweeps', checked, 'worst_wing_response_error', worst_wing.tolist(), 'worst_reference_kkt_gap', worst_gap, 'worst_terminal_relative_error', worst_field)
