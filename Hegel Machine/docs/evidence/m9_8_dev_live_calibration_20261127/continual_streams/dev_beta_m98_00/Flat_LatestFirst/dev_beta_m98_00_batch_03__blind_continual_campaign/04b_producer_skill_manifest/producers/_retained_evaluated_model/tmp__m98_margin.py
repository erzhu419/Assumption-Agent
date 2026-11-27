import importlib.util, json, pickle
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
log = json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text())
for entry in log:
    t = np.asarray(entry['facility_telemetry_trace'])
    print('prefix', entry['completed_batch_index'], 'mean', t.mean(axis=0).round(5), 'std', t.std(axis=0).round(5), 'last', t[-1].round(5))
for item in batch:
    s = item['study']
    t = s['aux_manifold_telemetry']
    print('current telemetry mean/std/last', t.mean(axis=0).round(5), t.std(axis=0).round(5), t[-1].round(5))
    axis = m._axis
    axis_inputs = []
    def capture(k, v, h):
        axis_inputs.append((k, v, h))
        return axis(k, v, h)
    m._axis = capture
    try:
        S, U = m._design(s)
    finally:
        m._axis = axis
    assert len(axis_inputs) == 2
    Z = s['Z_nuisance']
    projector = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (projector @ S)
    nx, ny = s['nx'], s['ny']
    max_conservation_error = 0.
    for (k, v, h), (length, grid) in zip(axis_inputs, [(s['Lx'], nx), (s['Ly'], ny)]):
        assert k.shape == v.shape == (grid,)
        np.testing.assert_allclose(h, length / (grid - 1), rtol=1e-14, atol=0)
        stencil = m._axis(k, v, h).toarray()
        ref = np.zeros_like(stencil)
        for i in range(grid):
            for step in (-1, 1):
                j = (i + step) % grid
                coeff = (k[i] + k[j]) / (2*h*h) + max(-step*v[i], 0.) / h
                ref[i, j] -= coeff
                ref[i, i] += coeff
        np.testing.assert_allclose(stencil, ref, rtol=1e-13, atol=1e-12)
        max_conservation_error = max(max_conservation_error, np.max(abs(stencil @ np.ones(grid))))
    cals = s['sensor_calibration_sweeps']; x = cals['ref_rosette']; y = cals['raw_rosette']
    F = np.column_stack((np.ones(len(x)), x))
    Q = np.column_stack([np.ones(len(x)), *x.T, *[x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)]])
    calibration = {'affine': 0., 'quadratic': 0.}
    max_gap = max_comp = max_field_error = 0.
    b = s['u_box']; budget = s['total_budget']; ridge = s['source_ridge']
    for fold in np.array_split(np.arange(len(x)), 5):
        train = np.setdiff1d(np.arange(len(x)), fold)
        for name, design in [('affine', F), ('quadratic', Q)]:
            coeff = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
            rms = np.sqrt(np.mean((design[fold] @ coeff - y[fold])**2, axis=0)).max()
            calibration[name] = max(calibration[name], rms)
    for raw in [s['y_obs_raw'], *[p['y_pulse_raw'] for p in item.get('campaign_diagnostics') or []]]:
        st = dict(s, y_obs_raw=raw)
        result = m.solve_batch(st); w = result['weights']; c = result['kkt_certificate']
        yy = m._calibrate(st); r = yy - Z @ (projector @ yy)
        e = A @ w - r; g = A.T @ e + ridge*w
        J = .5*(e@e + ridge*(w@w)); nu = c['nu_sum']
        lo = np.asarray(c['lam_lower']); hi = np.asarray(c['lam_upper'])
        dl = J - g@w + nu*budget + b*np.minimum(0., g-nu).sum()
        comp = max(np.max(abs(lo*w)), np.max(abs(hi*(b-w))))
        field_error = np.max(abs(result['terminal_field'] - np.tensordot(w, U, axes=(0, 0))))
        np.testing.assert_allclose(c['weights'], w, atol=1e-12, rtol=0)
        np.testing.assert_allclose([c['primal_objective'], c['dual_lower_bound']], [J, dl], atol=1e-11, rtol=0)
        assert abs(w.sum()-budget)<1e-10 and w.min()>-1e-10 and w.max()<b+1e-10
        assert lo.min()>=0 and hi.min()>=0 and np.max(abs(g-nu-lo+hi))<1e-10
        assert -1e-9 <= J-dl <= 2e-4 and comp<2e-4 and field_error<1e-12
        max_gap = max(max_gap, J-dl); max_comp = max(max_comp, comp); max_field_error = max(max_field_error, field_error)
    print('max periodic row-sum error:', max_conservation_error, 'worst calibration fold/channel RMSE:', calibration)
    print('worst observation/pulse KKT gap:', max_gap, 'complementarity error:', max_comp, 'terminal-field assembly error:', max_field_error)
