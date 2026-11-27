import importlib.util
import pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
path = root / 'beta_m97_c01_batch_01_solver.py'
spec = importlib.util.spec_from_file_location('reactor', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
public = pickle.loads((root / 'public_beta_m97_c01_batch_01.pkl').read_bytes())
for item in [public] if isinstance(public, dict) else public:
    s, gt = item['study'], item['gt']
    Z = s['Z_nuisance']
    G = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
    def residual(m): return m - Z @ np.linalg.solve(G, Z.T @ m)
    print('calibrated residual max error', np.max(abs(residual(solver._calibrated_observations(s)) - gt['r_y'])))
    base = path.read_text()
    for label, mask in [('both','sealed'),('x only',"sealed and d == 'x'"),('y only',"sealed and d == 'y'"),('neither','False')]:
        ns = {}
        exec(compile(base.replace('0. if sealed else', f'0. if {mask} else'), str(path), 'exec'), ns)
        S, _ = ns['_forward'](s)
        out = ns['solve_beta_m97_c01_batch_01'](s)
        errs = [np.max(abs(a-b)) for a,b in [(residual(S),gt['S_tilde']),(out['weights'],gt['w_opt']),(out['terminal_field'],gt['terminal_field_A'])]]
        cert = out['kkt_certificate']
        print(label, 'S/weights/field max errors', errs, 'nu',cert['nu_sum'],gt['cert_opt']['nu_sum'], 'gap',cert['primal_objective']-cert['dual_lower_bound'])
