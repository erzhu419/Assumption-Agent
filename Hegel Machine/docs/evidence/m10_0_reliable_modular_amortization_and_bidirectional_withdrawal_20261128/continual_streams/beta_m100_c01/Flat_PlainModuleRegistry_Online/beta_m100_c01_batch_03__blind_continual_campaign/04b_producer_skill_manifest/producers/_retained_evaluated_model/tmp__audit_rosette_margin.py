import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
sys.path.insert(0, str(root))
import upstream_transducer as up
from batch_solver import solve_batch

items = pickle.loads((root / 'public_batch.pkl').read_bytes())
terms = [t for degree in range(4) for t in itertools.combinations_with_replacement(range(4), degree)]
def polynomial(x):
    return np.column_stack([np.prod(x[:, t], axis=1) for t in terms])

def rational(x, j):
    own = x[:, j]
    nxt = x[:, (j+1) % 4]
    prev = x[:, (j-1) % 4]
    q = np.column_stack((own, own+nxt, own-nxt, own+prev, own-prev))
    return np.hstack([q**3 / (1+a*q**2) for a in (.25, 1.)])

for item in items:
    s = item['study']
    c = s['sensor_calibration_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']
    clean = np.max(np.abs(y), axis=1) < 8
    axis = np.count_nonzero(x, axis=1) <= 1
    amplitude = np.max(np.abs(x), axis=1)
    mixed = np.flatnonzero(~axis)
    outer = mixed[np.argsort(amplitude[mixed])[-max(1, len(mixed)//5):]]
    masks = {'axis rosette block': axis, 'largest mixed amplitudes': np.isin(np.arange(len(x)), outer)}
    maxima = {}
    for label, te in masks.items():
        tr = ~te
        fit = up._fit(dict(c, ref_rosette=x[tr], raw_rosette=y[tr]))
        yp = up._forward(x, fit)
        resid = np.linalg.norm(yp-y, axis=1)
        keep = tr & (resid <= 8*max(np.median(resid[tr]), 1e-8))
        test = te & clean
        D = polynomial(x)
        cubic = np.linalg.lstsq(D[keep], y[keep], rcond=None)[0]
        # Rediscover the dominant polynomial support on training rows only.
        screened = np.zeros_like(cubic)
        for j in range(4):
            active = np.abs(cubic[:, j]) > 0.01
            screened[active, j] = np.linalg.lstsq(D[keep][:, active], y[keep, j], rcond=None)[0]
        augmented = []
        for j in range(4):
            A = np.column_stack((up._design(x,j), rational(x,j)))
            p = np.linalg.lstsq(A[keep], y[keep,j], rcond=None)[0]
            augmented.append(A[test] @ p)
        preds = (yp[test], (D @ cubic)[test], (D @ screened)[test], np.column_stack(augmented))
        rms = [np.sqrt(np.mean(np.sum((p-y[test])**2, axis=1))) for p in preds]
        inv = np.max(np.linalg.norm(up._invert(y[test], fit)-x[test], axis=1))
        print(label, 'clean n', int(test.sum()), 'retained training', int(keep.sum()), 'RMS (cyclic/full cubic/fold-screened/rational sums-differences)', np.asarray(rms), 'max inverse', inv)
        maxima[label] = (rms[0], inv)
    print('study keys:', sorted(s), '| top-level item metadata keys:', sorted(set(item)-{'study'}))
    print('max blocked cyclic RMS and inverse:', np.max(list(maxima.values()), axis=0))
    assert all(rms < 1e-4 and inv < 3e-4 for rms, inv in maxima.values())
    out = solve_batch(s)
    cert = out['kkt_certificate']
    assert abs(cert['primal_objective']-cert['dual_lower_bound']) < 1e-9
