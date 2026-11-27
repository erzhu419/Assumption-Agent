import json
import pickle
import runpy
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
m = runpy.run_path(str(root / 'beta_m97_c01_batch_01_solver.py'))
archived = runpy.run_path('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py')
with (root / 'public_beta_m97_c01_batch_01.pkl').open('rb') as f:
    cases = pickle.load(f)
prior = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())[-1]
for case in cases:
    s, gt = case['study'], case['gt']
    for label, sweep in [('public', s['sensor_calibration_sweeps']), ('prefix', prior)]:
        banks, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
        for bank in range(4):
            xb, yb = x[banks == bank], y[banks == bank]
            errors = []
            for i in range(xb.size):
                train = np.arange(xb.size) != i
                keep = m['_clean_calibration'](xb[train], yb[train])
                g, o = m['_affine'](xb[train][keep], yb[train][keep])
                errors.append(abs(g * xb[i] + o - yb[i]))
            print(label, 'bank', bank, 'leave-one-reference-out max raw error', max(errors))
    out = m['solve_beta_m97_c01_batch_01'](s)
    z = s['Z_nuisance']
    gram = z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1])
    def project(a):
        return a - z @ np.linalg.solve(gram, z.T @ a)
    design, _ = m['_forward'](s)
    robin_design, _ = archived['_forward'](s)
    errors = dict(sealed_design=np.max(np.abs(project(design) - gt['S_tilde'])), robin_design=np.max(np.abs(project(robin_design) - gt['S_tilde'])), calibrated_target=np.max(np.abs(project(m['_calibrate'](s)) - gt['r_y'])), raw_target=np.max(np.abs(project(np.asarray(s['y_obs_raw']).reshape(-1)) - gt['r_y'])), weights=np.max(np.abs(out['weights'] - gt['w_opt'])), terminal=np.max(np.abs(out['terminal_field'] - gt['terminal_field_A'])))
    print('Public max errors:', errors)
    assert errors['sealed_design'] < 1e-12 and errors['weights'] < 1e-6 and errors['terminal'] < 1e-7
    for axis in ('x', 'y'):
        mat = m['_axis'](s[f'kappa_{axis}'], s[f'vel_{axis}'], 0.0, s[f'L{axis}'])
        boundary_sums = (mat @ np.ones(mat.shape[0]))[[0, -1]]
        print(axis, 'sealed boundary row sums:', boundary_sums)
        np.testing.assert_allclose(boundary_sums, 0.0, atol=1e-12)
