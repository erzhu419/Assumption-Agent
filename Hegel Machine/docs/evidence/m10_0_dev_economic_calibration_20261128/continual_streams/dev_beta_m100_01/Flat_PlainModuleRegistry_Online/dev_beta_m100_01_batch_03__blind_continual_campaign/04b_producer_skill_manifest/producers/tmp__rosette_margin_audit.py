import pickle
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, '/holdout_task')
from upstream_transducer import _basis, _fit, _invert
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    studies = [item['study'] for item in pickle.load(f)]

for s in studies:
    sweep = s['sensor_calibration_sweeps']
    x, y = sweep['ref_rosette'], sweep['raw_rosette']
    print('study and sweep metadata keys:', sorted(s), sorted(sweep))
    # Gross spikes are excluded from scoring, never from the inputs to the candidate fit.
    score = np.max(abs(y), axis=1) < 5
    norm = np.max(abs(x), axis=1)
    mixed = np.flatnonzero(np.count_nonzero(x, axis=1) > 1)
    splits = [(f'blocked-{i}', block) for i, block in enumerate(np.array_split(np.arange(len(x)), 5))]
    splits.append(('largest-mixed-amplitude', mixed[np.argsort(norm[mixed])[-len(mixed)//5:]]))
    q = np.column_stack([*x.T, *(sign*x[:, i] + x[:, j] for i, j in combinations(range(4), 2) for sign in (1, -1))])
    a = _basis(x)
    for name, block in splits:
        train = np.setdiff1d(np.arange(len(x)), block)
        test = block[score[block]]
        b = _fit(x[train], y[train])
        residual = np.linalg.norm(a[train] @ b - y[train], axis=1)
        inliers = train[residual <= max(8*np.median(residual), 1e-8)]
        # Profile rational response directions and strengths on training rows only.
        rational = []
        for unit in range(4):
            choices = []
            for j in range(q.shape[1]):
                for alpha in (1e-4, 1e-3, 0.01, 0.1, 0.3):
                    design = a / (1 + alpha*q[:, j:j+1]**2)
                    coef = np.linalg.lstsq(design[inliers], y[inliers, unit], rcond=None)[0]
                    resid = design[inliers] @ coef - y[inliers, unit]
                    choices.append((resid @ resid, design[test] @ coef))
            rational.append(min(choices, key=lambda c: c[0])[1])
        forward = np.max(abs(a[test] @ b - y[test]))
        inverse = np.max(abs(_invert(y[test], b) - x[test]))
        rational_error = np.max(abs(np.column_stack(rational) - y[test]))
        print(f'{name}: cubic-forward={forward:.4g} cubic-inverse={inverse:.4g} rational-forward={rational_error:.4g}')
        assert inverse < 1e-4, (name, inverse)
    # Compare actual inversion errors against the clean sweep noise floor, not a synthetic round trip.
    b = _fit(x, y)
    raw_obs = np.asarray(s['y_obs_raw']).reshape(-1, 4)
    obs = _invert(raw_obs, b)
    singular = np.linalg.svd(b[1:5], compute_uv=False)[-1]
    print(f'linear minimum singular value={singular:.5g}; max reference amplitude: observed={np.max(abs(obs)):.5g}, measured clean sweep={np.max(abs(x[score])):.5g}')
    out = solve_batch(s)
    cert = out['kkt_certificate']
    gap = abs(cert['primal_objective'] - cert['dual_lower_bound'])
    print(f'composed primal-dual gap={gap:.4g}')
    assert gap < 1e-8
