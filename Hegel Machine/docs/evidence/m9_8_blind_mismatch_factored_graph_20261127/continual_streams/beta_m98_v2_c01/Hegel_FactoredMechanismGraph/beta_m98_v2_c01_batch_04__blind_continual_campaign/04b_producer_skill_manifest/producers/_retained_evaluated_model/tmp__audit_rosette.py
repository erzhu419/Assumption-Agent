import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
from upstream_transducer import _design, _fit, _invert

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    studies = [item['study'] for item in pickle.load(f)]
print('study count', len(studies))
for st in studies:
    sweeps = st['sensor_calibration_sweeps']
    x, y = (np.asarray(sweeps[k]) for k in ('ref_rosette', 'raw_rosette'))
    print('study keys', sorted(st))
    a = _design(x)
    poly = np.empty_like(y)
    rival = np.empty_like(y)
    selected = np.zeros(y.shape, dtype=bool)
    for i in range(len(x)):
        train = np.arange(len(x)) != i
        b = _fit(x[train], y[train])
        poly[i] = a[i] @ b
        r = np.linalg.norm(y[train] - a[train] @ b, axis=1)
        usable = np.flatnonzero(train)[r <= max(12 * np.median(r), 1e-7)]
        for j in range(4):
            keep = b[:, j] != 0
            base = a[:, keep]
            z = y[usable, j]
            best_score = np.inf
            q_options = [x[:, j], x[:, j] + x[:, (j+1)%4], x[:, j] - x[:, (j+1)%4]]
            for q in q_options:
                for alpha in (0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.):
                    den = 1 + alpha * q*q
                    features = base / den[:, None]
                    fit = features[usable]
                    coef = np.linalg.lstsq(fit, z, rcond=None)[0]
                    leverage = np.sum(np.linalg.qr(fit, mode='reduced')[0]**2, axis=1)
                    press = np.mean(((fit @ coef - z) / (1 - leverage))**2)
                    if press < best_score:
                        best_score = press
                        rival[i, j] = features[i] @ coef
                        selected[i, j] = alpha > 0
    ep = np.linalg.norm(poly-y, axis=1)
    er = np.linalg.norm(rival-y, axis=1)
    clean = ep < 12*np.median(ep)
    high = np.max(abs(x), axis=1) > 1.4
    for label, use in [('nominal', clean), ('tail', clean & high)]:
        print(label, 'rows', use.sum(), 'polynomial/rational max', ep[use].max(), er[use].max(), 'RMS', np.sqrt(np.mean(ep[use]**2)), np.sqrt(np.mean(er[use]**2)), 'nonzero denominator selections', selected[use].sum())
    print('flagged sweep rows', np.flatnonzero(~clean).tolist())
    banks = sweeps['single_bank_sweeps']
    j = np.asarray(banks['bank_id'], dtype=int).reshape(-1)
    bx = np.eye(4)[j] * np.asarray(banks['ref_signal']).reshape(-1, 1)
    # Use only mixed-channel sweeps for this independent single-bank check.
    mixed = np.count_nonzero(x, axis=1) > 1
    b = _fit(x[mixed], y[mixed])
    discrepancy = ((_design(bx) @ b)[np.arange(len(j)), j] - np.asarray(banks['raw_readout']).reshape(-1))
    for bank in range(4):
        dr = discrepancy[j == bank]
        nominal = np.abs(dr) < 12 * np.median(np.abs(discrepancy))
        print('bank', bank, 'nominal max', np.max(np.abs(dr[nominal])), 'EMI count', (~nominal).sum())
        assert np.max(np.abs(dr[nominal])) < 2e-4
    assert ep[clean].max() < 2e-4
