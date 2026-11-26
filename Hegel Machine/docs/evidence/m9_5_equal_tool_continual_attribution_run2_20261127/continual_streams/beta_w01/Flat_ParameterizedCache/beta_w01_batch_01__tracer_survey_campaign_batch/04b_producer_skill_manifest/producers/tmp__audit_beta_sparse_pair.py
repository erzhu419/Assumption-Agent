import importlib.util, json, pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'beta_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
solve = m.solve_beta_w01_batch_01
full_responses = m._responses
history = json.loads(m._HISTORY.read_text())
for case in pickle.load(open(root+'public_beta_w01_batch_01.pkl', 'rb')):
    s, gt = case['study'], case['gt']
    c = s['sensor_calibration_sweeps']
    row = s['row_bank_id']
    responses = full_responses(s)
    m._responses = lambda _: responses
    for b in range(4):
        idx = np.flatnonzero(c['bank_id'] == b)
        idx = idx[np.argsort(c['ref_signal'][idx])]
        hg, ho = m._historical_calibration(history, b)
        x, y = c['ref_signal'][idx], c['raw_readout'][idx]
        mismatch = np.abs(np.diff(y-hg*x))
        worst = np.zeros(3)
        for i in range(len(idx)-1):
            # Apply the historical gain with an offset measured in this batch.
            offset = np.mean((y-hg*x)[i:i+2])
            raw = s['y_obs_raw'].copy()
            raw[row == b] = (raw[row == b]-offset)/hg
            clean = c['bank_id'] != b
            sweep = {k: v.copy() for k, v in c.items()}
            sweep['raw_readout'][~clean] = sweep['ref_signal'][~clean]
            out = solve(dict(s, y_obs_raw=raw, sensor_calibration_sweeps=sweep))
            u = gt['terminal_field_mid_budget']
            worst = np.maximum(worst, [np.max(abs(out['weights_by_budget']-gt['weights_by_budget'])), np.linalg.norm(out['terminal_field_mid_budget']-u)/np.linalg.norm(u), max(abs(p['primal_objective']-q['primal_objective']) for p,q in zip(out['kkt_certificates'],gt['certs_by_budget']))])
        print('bank', b, 'archived gain/offset', (hg,ho), 'minimum pair span', np.min(np.diff(x)), 'max pair residual difference', mismatch.max(), 'fallback max weight abs / field rel / objective abs', worst)
