import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root/'beta_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}
exec((root/'validate_public_schema.py').read_text().split('\n_d = ')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root/'public_beta_m97_c03_batch_01.pkl').open('rb') as f:
    cases = pickle.load(f)
for case in cases:
    s, g = case['study'], case['gt']
    sweeps = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}
    for b in range(4):
        worst = np.zeros(3)
        for j in np.flatnonzero(sweeps['bank_id'] == b):
            for spike in (-3.01, 3.01):
                sweeps_emi = dict(sweeps, raw_readout=sweeps['raw_readout'].copy())
                sweeps_emi['raw_readout'][j] += spike
                out = m.solve_beta_m97_c03_batch_01(dict(s, sensor_calibration_sweeps=sweeps_emi))
                cert = verify(g['S_tilde'], g['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
                assert cert['valid'], (b, j, spike, cert)
                metrics = [cert['gap'], np.max(np.abs(out['weights']-g['w_opt'])), np.linalg.norm(out['terminal_field']-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A'])]
                worst = np.maximum(worst, metrics)
        print(f'bank={b} worst signed EMI reference KKT gap={worst[0]:.3g}; weight max error={worst[1]:.3g}; field relative L2={worst[2]:.3g}')
    for axis in ('x','y'):
        k = np.asarray(s['kappa_'+axis]); h = s['L'+axis]/(len(k)-1)
        A = m._axis(k, s['vel_'+axis], 0., s['L'+axis]).toarray()
        ends = np.zeros_like(A[[0,-1]])
        ends[0,:2] = [k[0]/h**2, -k[0]/h**2]
        ends[1,-2:] = [-k[-1]/h**2, k[-1]/h**2]
        np.testing.assert_allclose(A[[0,-1]], ends, rtol=1e-14, atol=0)
        print(f'{axis}: full sealed endpoint rows match prescribed coefficients')
