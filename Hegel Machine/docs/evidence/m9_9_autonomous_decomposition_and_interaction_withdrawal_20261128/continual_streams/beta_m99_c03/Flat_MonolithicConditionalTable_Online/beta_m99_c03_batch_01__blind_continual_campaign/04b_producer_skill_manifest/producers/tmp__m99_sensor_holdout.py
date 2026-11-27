import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = item['study']
    S, U = m._forward(s)
    sweeps = s['sensor_calibration_sweeps']
    bank = sweeps['single_bank_sweeps']
    xf, yf = sweeps['ref_rosette'], sweeps['raw_rosette']
    coef = []
    for j in range(4):
        take = bank['bank_id'] == j
        coef.append(np.polyfit(np.r_[xf[:, j], bank['ref_signal'][take]], np.r_[yf[:, j], bank['raw_readout'][take]], 1))
    gain, offset = np.array(coef).T
    observed = np.array([p['y_pulse_raw'] for p in item['campaign_diagnostics']])
    expected = np.array([S @ p['w_pulse'] for p in item['campaign_diagnostics']])
    active = np.array([m._calibrated_observations({**s, 'y_obs_raw': raw}) for raw in observed])
    pooled = (((observed.reshape(len(observed), -1, 4) - offset) / gain)).reshape(observed.shape)
    def worst(v):
        e = (v - expected).reshape(len(observed), len(s['mu_shifts']), -1, 4)
        return np.sqrt(np.mean(e**2, axis=2)).max()
    print('Worst tracer pulse/shift/bank RMS, rosette-only versus pooled bench sweeps:', worst(active), worst(pooled))

    y = m._calibrated_observations(s)
    n_sensors = len(s['sensor_ij'])
    assert n_sensors % 4 == 0
    base = m.solve_batch(s)['weights']
    gaps, residuals, weight_changes, design_ratios = [], [], [], []
    for start in range(0, n_sensors, 4):
        keep_sensor = np.ones(n_sensors, dtype=bool)
        keep_sensor[start:start+4] = False
        keep = np.tile(keep_sensor, len(s['mu_shifts']))
        Z = s['Z_nuisance'][keep]
        obs = np.column_stack((S[keep], y[keep]))
        obs -= Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T @ obs)
        D, r = obs[:, :-1], obs[:, -1]
        sing = np.linalg.svd(D, compute_uv=False)
        design_ratios.append(sing[-1]/sing[0])
        h = {**s, 'sensor_ij': s['sensor_ij'][keep_sensor], 'Z_nuisance': Z, 'y_obs_raw': s['y_obs_raw'][keep]}
        if 'row_bank_id' in s:
            h['row_bank_id'] = s['row_bank_id'][keep]
        out = m.solve_batch(h)
        w, cert = out['weights'], out['kkt_certificate']
        e = D @ w - r
        obj = (e @ e + s['source_ridge']*(w @ w))/2
        g = D.T @ e + s['source_ridge']*w
        err = g - cert['nu_sum'] - np.array(cert['lam_lower']) + np.array(cert['lam_upper'])
        gaps.append(abs(obj - cert['dual_lower_bound']))
        residuals.append(np.max(np.abs(err)))
        weight_changes.append(np.linalg.norm(w-base)/np.linalg.norm(base))
        assert abs(w.sum()-s['total_budget']) < 1e-10 and min(w) >= -1e-12 and max(w) <= s['u_box']+1e-12
        assert gaps[-1] < 1e-9 and residuals[-1] < 1e-9
    print('Held-out rosettes:', len(gaps), 'worst KKT gap:', max(gaps), 'worst stationarity residual:', max(residuals), 'minimum projected singular ratio:', min(design_ratios), 'maximum relative weight change:', max(weight_changes))
