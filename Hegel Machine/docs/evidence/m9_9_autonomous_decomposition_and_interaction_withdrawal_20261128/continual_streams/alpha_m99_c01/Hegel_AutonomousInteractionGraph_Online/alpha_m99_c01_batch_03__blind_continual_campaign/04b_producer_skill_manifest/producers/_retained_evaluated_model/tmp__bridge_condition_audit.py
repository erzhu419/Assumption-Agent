import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_bridge, _inverse

j = np.arange(4)
k = (j+1)%4
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    d = item['dataset']; c = d['calibration_load_cells']; x = c['true_channel_loads']
    b = _fit_bridge(x, c['bridge_raw'])
    operating = np.vstack([_inverse(d[g][f], b).reshape(-1,4) for g in ('train','test') for f in ('past_channel_raw','hold_channel_raw')])
    points = np.vstack((x, operating))
    lo, hi = points.min(0), points.max(0)
    pending = [(lo, hi, 0)]
    certified = np.inf; sampled = np.inf; count = 0; unresolved = 0
    while pending:
        low, high, depth = pending.pop()
        mid = (low+high)/2; rad = (high-low)/2
        jac = b[1:5].T.copy()
        jac[j,j] += 3*b[5]*mid**2 + b[6]*mid[k]
        jac[j,k] += b[6]*mid
        ev = np.linalg.eigvalsh((jac+jac.T)/2)[0]
        sampled = min(sampled, ev)
        # Use the exact diagonal minimum on the sub-box plus absolute
        # off-diagonal perturbation radii to bound the eigenvalue from below.
        closest = np.maximum(low,0)+np.minimum(high,0)
        sq_min = np.minimum(3*b[5]*closest**2, 3*b[5]*np.maximum(low**2,high**2))
        diag_loss = 3*b[5]*mid**2 - sq_min + np.abs(b[6])*rad[k]
        edge = np.abs(b[6])*rad/2
        bound = ev - np.max(diag_loss+edge+np.roll(edge,1))
        if bound > 0.02 or depth == 16:
            certified = min(certified,bound); count += 1; unresolved += bound <= 0
        else:
            i = np.argmax(rad)
            p = high.copy(); p[i] = mid[i]
            q = low.copy(); q[i] = mid[i]
            pending.extend(((low,p,depth+1),(q,high,depth+1)))
    print(f'operating and calibration box: symmetric Jacobian lower bound={certified:.9g}; minimum center eigenvalue={sampled:.9g}; sub-boxes={count}; unresolved={unresolved}')
    assert unresolved == 0
