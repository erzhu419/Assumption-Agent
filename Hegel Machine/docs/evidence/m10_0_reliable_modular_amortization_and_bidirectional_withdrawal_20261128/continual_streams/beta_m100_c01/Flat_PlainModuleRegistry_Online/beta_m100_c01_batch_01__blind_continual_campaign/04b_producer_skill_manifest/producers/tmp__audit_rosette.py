import importlib.util
import pickle
from itertools import combinations
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    st = item['study']; sw = st['sensor_calibration_sweeps']
    x = np.asarray(sw['ref_rosette']); y = np.asarray(sw['raw_rosette'])
    print('Calibration rows:', len(x), 'study keys:', sorted(st))
    print('Single-bank sweep layout:', [(k, np.shape(v)) for k, v in sw['single_bank_sweeps'].items()] if isinstance(sw.get('single_bank_sweeps'), dict) else type(sw.get('single_bank_sweeps')).__name__)
    paired = [x[:, i] * x[:, j] for i, j in combinations(range(4), 2)]
    affine = np.column_stack((np.ones(len(x)), x))
    models = {'affine': affine,
              'quadratic': np.column_stack((affine, x*x, *paired)),
              'saturating': np.column_stack((affine, x/(1+x*x), *[(x[:, i]-x[:, j])/(1+(x[:, i]-x[:, j])**2) for i,j in combinations(range(4), 2)]))}
    # Hold out whole amplitude bands to test extrapolation, not adjacent sweep rows.
    bands = np.array_split(np.argsort(np.linalg.norm(x, axis=1)), 5)
    weights = solver.solve_batch(st)['weights']
    by_model = {k: [] for k in models}
    margins = []
    for test in bands:
        train = np.setdiff1d(np.arange(len(x)), test)
        for name, a in models.items():
            beta = np.linalg.lstsq(a[train], y[train], rcond=None)[0]
            by_model[name].append(np.sqrt(np.mean((a[test] @ beta-y[test])**2, axis=0)).max())
        slim = dict(st, sensor_calibration_sweeps=dict(sw, ref_rosette=x[train], raw_rosette=y[train]))
        margins.append(np.max(np.abs(solver.solve_batch(slim)['weights']-weights)))
    print('Worst held-out amplitude-band/channel RMS:', {k: max(v) for k,v in by_model.items()})
    print('Max weight change with a calibration amplitude band removed:', max(margins))
    S, _ = solver._transport(st)
    Z = np.asarray(st['Z_nuisance'])
    Q = np.linalg.qr(Z)[0]
    ds = item.get('campaign_diagnostics') or []
    if ds:
        W = np.column_stack([d['w_pulse'] for d in ds])
        Y = np.column_stack([solver._calibrate(dict(st, y_obs_raw=d['y_pulse_raw'])) for d in ds])
        R = S @ W - Y
        for name, e in [('unprojected', R), ('orthogonally profiled', R-Q@(Q.T@R))]:
            # Each tensor cell pools sensors of one rosette channel and shift, for one pulse.
            v = e.reshape(len(st['mu_shifts']), -1, 4, len(ds))
            rms = np.sqrt(np.mean(v*v, axis=1)).max()
            print(name, 'worst shift/channel/pulse RMS:', rms)
            assert rms < 5e-6
    assert max(by_model['affine']) < 5e-5 and max(margins) < 1e-4
