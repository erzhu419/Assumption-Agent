import importlib.util
import pickle
import time
from pathlib import Path
import numpy as np

path = Path('/holdout_task/thermal_sink_solver.py')
spec = importlib.util.spec_from_file_location(path.stem, path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with path.with_name('public_thermal_cases.pkl').open('rb') as f:
    cases = pickle.load(f)

for i, case in enumerate(cases):
    m, truth = case['measurement'], case['gt']
    n, sigma = m['N'], m['sigma_m']
    ii, jj = (np.indices((n, n)) + .5) / n
    factor = solver._factor_operator(m['kappa'], m['gamma_robin'])
    sensor = tuple(m['sensor_ij'].T)
    y = m['T_obs']
    def field(xy, a):
        q = (a/n**2) * np.exp(-((ii-xy[0])**2 + (jj-xy[1])**2)/(2*sigma**2))
        return factor.solve(q.ravel()).reshape(n,n)[sensor]
    order = np.lexsort(m['sensor_ij'].T[::-1])
    fold_ids = np.empty(len(y), dtype=int)
    fold_ids[order] = np.arange(len(y)) % 4
    prediction = np.empty_like(y)
    worst = np.zeros(3)
    for fold in range(4):
        keep = fold_ids != fold
        p = solver.localize_thermal_source(dict(m, sensor_ij=m['sensor_ij'][keep], T_obs=y[keep]))
        prediction[~keep] = (field(p['source_xy'], p['amplitude']) + p['junction_offset'])[~keep]
        worst = np.maximum(worst, [np.linalg.norm(np.subtract(p['source_xy'], truth['source_xy'])), abs(p['amplitude']/truth['amplitude']-1), abs(p['junction_offset']-truth['junction_offset'])])
    r = y-prediction
    med = np.median(r)
    scale = 1.4826*np.median(abs(r-med))
    bad = abs(r-med) > 6*scale
    print('public', i, 'held-out residual MAD scale', scale, 'large held-out residuals', int(bad.sum()), 'worst training-fold parameter errors', worst, flush=True)
    assert np.all(worst <= [.035,.08,.008])
    assert scale < .004 and bad.sum() <= 5
    # Exercise the expensive near-wall branch on the same real geometry and
    # conductivity, preserving its real residual pattern and default threading.
    xy = (.006, .74)
    real_residual = y - field(truth['source_xy'], truth['amplitude']) - truth['junction_offset']
    edge = dict(m, T_obs=field(xy, 24.) + truth['junction_offset'] + real_residual)
    start = time.perf_counter()
    p = solver.localize_thermal_source(edge)
    elapsed = time.perf_counter()-start
    print('near wall, default threading', i, 'seconds', elapsed, 'position error', np.linalg.norm(np.subtract(p['source_xy'],xy)), 'relative amplitude error', abs(p['amplitude']/24.-1), flush=True)
    assert elapsed < 15
