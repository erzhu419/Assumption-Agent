import sys, pickle
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_bridge, _forward, _inverse
from batch_solver import solve_batch
# Reuse the previously audited fitters without executing their validation loop.
with open('/tmp/audit_bridge.py') as f:
    exec(f.read().split('\nfor item in pickle.load', 1)[0])
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    d = item['dataset']; c = d['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']
    mixed = np.count_nonzero(x, axis=1) > 1
    groups = [('outer mixed shell', mixed & (np.max(abs(x),axis=1) > 1.4))]
    for j in range(4):
        k = (j+1)%4
        groups.append((f'large {j}-{k} contrast', mixed & (abs(x[:,j]-x[:,k]) >= np.quantile(abs(x[mixed,j]-x[mixed,k]), .75))))
    for name,test in groups:
        cal = dict(c,true_channel_loads=x[~test],bridge_raw=y[~test]); a = _fit_bridge(cal); b,_,_ = shared(cal); _,g,_ = shared(cal,True)
        no_cross = a.copy(); no_cross[:,6] = 0
        print(name, int(test.sum()), 'forward max [no interaction ablation, cyclic, shared, rational]', [float(np.max(abs(z-y[test]))) for z in (_forward(x[test],no_cross),_forward(x[test],a),_forward(x[test],b),g(x[test]))], 'inverse max [cyclic,shared]', [float(np.max(abs(_inverse(y[test],p)-x[test]))) for p in (a,b)])
    assert all(np.isfinite(v).all() for v in solve_batch(d).values())
