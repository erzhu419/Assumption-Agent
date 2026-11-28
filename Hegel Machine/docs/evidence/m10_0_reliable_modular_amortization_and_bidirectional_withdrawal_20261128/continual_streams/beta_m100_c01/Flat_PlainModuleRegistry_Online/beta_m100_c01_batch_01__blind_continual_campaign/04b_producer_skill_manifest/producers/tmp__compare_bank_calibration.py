import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    st = item['study']
    sw = st['sensor_calibration_sweeps']
    x, y = sw['ref_rosette'], sw['raw_rosette']
    single = sw['single_bank_sweeps']
    S, _ = solver._transport(st)
    ds = item['campaign_diagnostics']
    pulse_x = np.stack([(S @ p['w_pulse']).reshape(-1, 4) for p in ds])
    pulse_y = np.stack([p['y_pulse_raw'].reshape(-1, 4) for p in ds])
    bands = np.array_split(np.argsort(np.linalg.norm(x, axis=1)), 5)
    for model in ('full', 'diagonal', 'joint_diagonal'):
        def fit(rows):
            if model == 'full':
                return np.linalg.lstsq(np.c_[np.ones(len(rows)), x[rows]], y[rows], rcond=None)[0]
            b = np.zeros((5, 4))
            for j in range(4):
                u, v = x[rows, j], y[rows, j]
                if model == 'joint_diagonal':
                    mask = single['bank_id'] == j
                    u = np.r_[u, single['ref_signal'][mask]]
                    v = np.r_[v, single['raw_readout'][mask]]
                b[[0, j+1], j] = np.linalg.lstsq(np.c_[np.ones(len(u)), u], v, rcond=None)[0]
            return b
        rms = []
        for test in bands:
            b = fit(np.setdiff1d(np.arange(len(x)), test))
            e = x[test] @ b[1:] + b[0] - y[test]
            rms.append(np.sqrt(np.mean(e*e, axis=0)).max())
        b = fit(np.arange(len(x)))
        residual = pulse_x @ b[1:] + b[0] - pulse_y
        e = np.linalg.solve(b[1:].T, residual.reshape(-1,4).T).T.reshape(len(ds), len(st['mu_shifts']), -1, 4)
        print(model, 'held-out sweep worst RMS', max(rms), 'independent tracer worst / pooled RMS', np.sqrt(np.mean(e*e,axis=2)).max(), np.sqrt(np.mean(e*e)), 'tracer per-pulse RMS', np.sqrt(np.mean(e*e,axis=(1,2,3))))
