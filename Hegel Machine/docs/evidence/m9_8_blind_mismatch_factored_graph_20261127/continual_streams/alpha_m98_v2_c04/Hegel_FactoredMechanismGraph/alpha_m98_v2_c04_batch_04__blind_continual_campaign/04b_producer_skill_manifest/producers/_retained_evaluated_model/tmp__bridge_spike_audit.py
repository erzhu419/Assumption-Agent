import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
cal = ds['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
a = u._basis(x)
rows = np.arange(len(x))
full = u._fit_bridge(x, y)
r = np.linalg.norm(y - a @ full, axis=1)
clean = r <= 6*np.median(r)
spikes = rows[~clean]
assert len(spikes) > 0
op = np.concatenate([ds[k][f].reshape(-1,4) for k in ('train','test') for f in u._LOADS])
ref = u._invert_bridge(op, full)
print('Detected spike rows:', spikes.tolist())
for j in spikes:
    keep = rows != j
    delta = u._invert_bridge(op, u._fit_bridge(x[keep], y[keep])) - ref
    print('Remove spike', j, 'max operational channel/total shift', np.max(abs(delta)), np.max(abs(delta.sum(1))))
folds = [rows[16+i::4] for i in range(4)] + [rows[4*i:4*i+4] for i in range(4)] + [rows[(rows>=16)&(np.max(abs(x),axis=1)>1.45)]]
worst = {'quadratic_raw': 0., 'cubic_raw': 0.}
for hold in folds:
    train = ~np.isin(rows, hold)
    test = hold[clean[hold]]
    coef = u._fit_bridge(x[train], y[train])
    residual = np.linalg.norm(y[train] - a[train] @ coef, axis=1)
    keep = rows[train][residual <= 6*np.median(residual)]
    quad = np.linalg.lstsq(a[keep,:15], y[keep], rcond=None)[0]
    worst['quadratic_raw'] = max(worst['quadratic_raw'], np.max(np.linalg.norm(y[test] - a[test,:15] @ quad, axis=1)))
    worst['cubic_raw'] = max(worst['cubic_raw'], np.max(np.linalg.norm(y[test] - a[test] @ coef, axis=1)))
print('Same-fold worst raw residual norms:', worst)
