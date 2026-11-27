import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'beta_m96_w01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_beta_m96_w01_s00.pkl','rb') as f:
    cases = pickle.load(f)
worst = np.zeros((3, 3))
count = 0
for c in cases:
    s, gt = c['study'], c['gt']
    sw = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    bank, ref = sw['bank_id'], sw['ref_signal']
    for wing, affected in enumerate(((0,1), (2,3), (0,1,2,3))):
        for positions in ((0,4,7,11), (0,2,9,11), (1,3,8,10)):
            for spike in (None, 0, 1, 2, 3):
                for sign in ((1,) if spike is None else (-1,1)):
                    keep = np.ones(len(ref), dtype=bool)
                    raw = sw['raw_readout'].copy()
                    for b in affected:
                        idx = np.flatnonzero(bank == b)
                        idx = idx[np.argsort(ref[idx])]
                        retained = idx[list(positions)]
                        keep[idx] = False
                        keep[retained] = True
                        if spike is not None:
                            raw[retained[spike]] += sign * 0.1 * (-1)**b
                    trial = dict(s, sensor_calibration_sweeps=dict(bank_id=bank[keep], ref_signal=ref[keep], raw_readout=raw[keep]))
                    out = m.solve_beta_m96_w01_s00(trial)
                    cert = out['kkt_certificate']
                    ew = np.max(np.abs(out['weights']-gt['w_opt']))
                    ef = np.linalg.norm(out['terminal_field']-gt['terminal_field_A'])/np.linalg.norm(gt['terminal_field_A'])
                    worst[wing] = np.maximum(worst[wing], [ew, ef, cert['primal_objective']-cert['dual_lower_bound']])
                    count += 1
print(f'{count} sparse-sweep cases; worst [weight abs error, field rel error, KKT gap]:')
for name, row in zip(('wing A','wing B','both wings'), worst):
    print(name, ' '.join(f'{v:.3e}' for v in row))
assert np.max(worst) < 1e-6
