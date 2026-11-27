import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'beta_m97_c02_batch_01_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
for item in pickle.load(open(root+'public_beta_m97_c02_batch_01.pkl','rb')):
    s, gt = item['study'], item['gt']
    sweeps = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}
    b = sweeps['bank_id']
    x, y = sweeps['ref_signal'], sweeps['raw_readout']
    rb = np.asarray(s['row_bank_id'])
    raw = np.asarray(s['y_obs_raw'])
    z = s['Z_nuisance']
    gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
    def project(v):
        return v - z @ np.linalg.solve(gram, z.T @ v)
    worst_heldout = worst_projected = 0.0
    for bank in range(4):
        heldout = projected = 0.0
        for i in np.flatnonzero(b == bank):
            params = solver._calibration_parameters({k: v[np.arange(b.size) != i] for k,v in sweeps.items()})
            gain, offset = params[bank]
            heldout = max(heldout, abs(y[i] - (gain*x[i] + offset)))
            gain, offset = params[rb].T
            d = project((raw-offset)/gain) - gt['r_y']
            projected = max(projected, np.max(np.abs(d[rb == bank])))
        bypass = np.max(np.abs((project(raw)-gt['r_y'])[rb == bank]))
        print(f'bank {bank}: worst held-out sweep residual {heldout:.3e}; worst projected target error {projected:.3e}; uncalibrated error {bypass:.3e}')
        worst_heldout = max(worst_heldout, heldout)
        worst_projected = max(worst_projected, projected)
    assert worst_heldout < 1e-5 and worst_projected < 1e-5
    design, _ = solver._forward(s)
    design_error = np.linalg.norm(project(design)-gt['S_tilde'])/np.linalg.norm(gt['S_tilde'])
    print(f'full-stack discrete operator relative L2 error: {design_error:.3e}')
    assert design_error < 1e-12
