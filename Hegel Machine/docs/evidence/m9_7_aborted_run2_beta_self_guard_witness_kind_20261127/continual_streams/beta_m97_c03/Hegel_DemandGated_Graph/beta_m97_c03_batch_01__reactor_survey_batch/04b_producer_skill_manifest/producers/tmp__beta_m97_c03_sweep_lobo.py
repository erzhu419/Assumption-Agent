import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class NoArchive:
    def open(self, *args, **kwargs):
        raise AssertionError('Clean current sweeps must not consult the archive')

m._ARCHIVE = NoArchive()
with open('/holdout_task/public_beta_m97_c03_batch_01.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    s, gt = case['study'], case['gt']
    sweeps = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    bank, x, raw = (sweeps[k] for k in ('bank_id', 'ref_signal', 'raw_readout'))
    worst = np.zeros((4, 4))
    for b in range(4):
        idx = np.flatnonzero(bank == b)
        for held in idx:
            pool = idx[idx != held]
            for selection in (pool, pool[:4], pool[-4:]):
                keep = bank != b
                keep[selection] = True
                sweep = {k: v[keep] for k, v in sweeps.items()}
                probe = dict(s, sensor_calibration_sweeps=sweep)
                recovered = m._calibrate(dict(probe, y_obs_raw=raw[held:held+1], row_bank_id=bank[held:held+1]))[0]
                out = m.solve_beta_m97_c03_batch_01(probe)
                w = np.asarray(out['weights'])
                d = gt['S_tilde'].T @ (gt['S_tilde'] @ w - gt['r_y']) + s['source_ridge']*w - out['kkt_certificate']['nu_sum']
                worst[b] = np.maximum(worst[b], [abs(recovered-x[held]), np.max(np.abs(w-gt['w_opt'])), np.max(np.abs(out['terminal_field']-gt['terminal_field_A'])), d@w-s['u_box']*np.minimum(d,0).sum()])
        print('bank', b, 'max held-out reference, weight, field, KKT gap:', *worst[b])
    assert np.all(worst < [1e-5, 1e-5, 1e-5, 2e-4]), worst
