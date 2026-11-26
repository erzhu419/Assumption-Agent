import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'beta_w03_batch_07_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for item in pickle.loads((root / 'public_beta_w03_batch_07.pkl').read_bytes()):
    s, gt = item['study'], item['gt']
    sweep = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}
    bank, x, y = (sweep[k] for k in ('bank_id','ref_signal','raw_readout'))
    g, o = mod._ARCHIVE[1, bank].T
    clean = np.abs(y - g*x - o) <= 2e-6
    for name, omit in [('public', np.zeros(len(bank), dtype=bool)), ('hold_out_bank_2_anchor', (bank==2)&clean), ('hold_out_bank_3_anchor', (bank==3)&clean), ('hold_out_both_anchors', (bank>=2)&clean)]:
        out = mod.solve_beta_w03_batch_07(dict(s, sensor_calibration_sweeps={k:v[~omit] for k,v in sweep.items()}))
        w, f = out['weights'], out['terminal_field']
        D, z = gt['S_tilde'], gt['r_y']
        grad = D.T @ (D @ w-z) + s['source_ridge']*w
        nu = out['kkt_certificate']['nu_sum']
        gap = grad@w - nu*s['total_budget'] - s['u_box']*np.minimum(0, grad-nu).sum()
        errors = [np.linalg.norm(a-b)/np.linalg.norm(b) for a,b in [(w,gt['w_opt']), (f,gt['terminal_field_A'])]]
        print(name, 'omitted', omit.sum(), 'weight/field relative L2', *errors, 'reference KKT gap',gap)
