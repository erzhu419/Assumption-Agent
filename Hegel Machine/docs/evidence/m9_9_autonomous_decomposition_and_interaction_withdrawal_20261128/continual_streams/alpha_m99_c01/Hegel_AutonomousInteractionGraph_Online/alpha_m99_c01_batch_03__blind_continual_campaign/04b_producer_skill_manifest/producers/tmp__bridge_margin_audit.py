import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_bridge, _forward, _inverse

for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    d = item['dataset']; c = d['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']
    b = _fit_bridge(x, y)
    ordinary = np.linalg.norm(y, axis=1) < 10
    # Detect coefficient or prediction drift caused by the actual spike rows.
    clean_fit = _fit_bridge(x[ordinary], y[ordinary])
    probes = [d[group][field].reshape(-1, 4) for group in ('train', 'test') for field in ('past_channel_raw', 'hold_channel_raw')]
    raw = np.vstack(probes)
    recovered = _inverse(raw, b)
    spike_effect = float(np.max(np.abs(recovered - _inverse(raw, clean_fit))))
    print('actual spikes: maximum change in operating load recovery', spike_effect)
    assert spike_effect < 1e-5
    # Certify the symmetric Jacobian is positive definite throughout the
    # observed calibration and operating box, not just at sampled inputs.
    bounds = np.vstack((x, recovered)); lo, hi = bounds.min(0), bounds.max(0)
    nearest = np.maximum(lo, 0) + np.minimum(hi, 0)
    nxt = (np.arange(4) + 1) % 4
    lower = np.diag(b[1:5]) + 3*b[5]*nearest**2 + np.minimum(b[6]*lo[nxt], b[6]*hi[nxt])
    assert np.all(b[5] >= 0)
    jlo = b[1:5].T.copy(); jhi = jlo.copy()
    jlo[np.arange(4), nxt] += np.minimum(b[6]*lo, b[6]*hi)
    jhi[np.arange(4), nxt] += np.maximum(b[6]*lo, b[6]*hi)
    off = np.maximum(np.abs((jlo+jlo.T)/2), np.abs((jhi+jhi.T)/2))
    np.fill_diagonal(off, 0)
    margin = np.min(lower-off.sum(axis=1))
    print('symmetric Jacobian lower eigenvalue bound over observed box', margin)
    assert margin > 0
