import itertools as it
import pickle
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _basis, _fit_bridge, _inverse
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
ds = item['dataset']
cal = ds['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
axis = np.count_nonzero(x, axis=1) == 1
edge = np.max(np.abs(x), axis=1) >= 1.4
# These three gross spikes were identified in the previous leave-one-out audit.
# Mask them only when scoring; leave them in training whenever in the fold.
scorable = ~np.isin(np.arange(len(x)), [13,14,15])
mixed_cubics = [p for p in it.combinations_with_replacement(range(4),3) if p[0] != p[-1]]
I = np.eye(4)
qvecs = [*I, *(I[i]+s*I[j] for i,j in it.combinations(range(4),2) for s in (-1,1))]
configs = [(q,a) for q in qvecs for a in (.1,.5)]

def sparsity(b):
    return [np.flatnonzero(b[5:,j])+5 for j in range(4)]

def rational_basis(z, cfg):
    q,a = cfg
    return _basis(z) / (1+a*(z@q)**2)[:,None]

def cubic_basis(z):
    return np.column_stack((_basis(z), *(np.prod(z[:,p],axis=1) for p in mixed_cubics)))

def clean_fit(indices):
    a,z = x[indices], y[indices]
    b = _fit_bridge(a,z)
    e = np.linalg.norm(_basis(a)@b-z,axis=1)
    ok = e <= max(6*np.median(e), 1e-9)
    return b,a[ok],z[ok]

def rational_fit(b,z,v,cfg):
    # Add a rational correction on each channel's supported nonlinear terms.
    # Difference from the polynomial factor avoids a duplicate numerator.
    base = _basis(z)
    change = rational_basis(z,cfg)-base
    blocks = sparsity(b)
    coefs = [np.linalg.lstsq(np.column_stack((base[:,:5],base[:,cols],change[:,cols])),v[:,j],rcond=None)[0] for j,cols in enumerate(blocks)]
    def forward(t):
        a = _basis(t)
        delta = rational_basis(t,cfg)-a
        return np.column_stack([np.column_stack((a[:,:5],a[:,cols],delta[:,cols]))@c for cols,c in zip(blocks,coefs)])
    return forward

def score_forward(e):
    return np.mean(e*e)

def inverse_metrics(forward, initial, rows):
    err = []
    for u,v,t in zip(initial,y[rows],x[rows]):
        fit = least_squares(lambda q:forward(q[None,:])[0]-v, u, gtol=1e-13, xtol=1e-13, ftol=1e-13)
        assert np.max(np.abs(fit.fun)) < 1e-8
        err.append(fit.x-t)
    return np.array(err)

rng = np.random.default_rng(495)
all_idx = np.arange(len(x))
folds = [(f'fold{i}', fold) for i,fold in enumerate(np.array_split(rng.permutation(all_idx),5))]
folds += [('axis', all_idx[axis]), ('mixed boundary', all_idx[~axis & edge])]
worst = np.zeros((3,2))
for name,outer in folds:
    train = np.setdiff1d(all_idx,outer)
    rows = outer[scorable[outer]]
    inner_score = np.zeros(len(configs))
    base_score = 0.
    for inner in np.array_split(rng.permutation(train),3):
        b,z,v = clean_fit(np.setdiff1d(train,inner))
        inner = inner[scorable[inner]]
        base_score += score_forward(_basis(x[inner])@b-y[inner])
        for ii,cfg in enumerate(configs):
            inner_score[ii] += score_forward(rational_fit(b,z,v,cfg)(x[inner])-y[inner])
    b,z,v = clean_fit(train)
    initial = _inverse(y[rows],b)
    c = np.linalg.lstsq(cubic_basis(z),v,rcond=None)[0]
    cfg = configs[np.argmin(inner_score)]
    errors = [initial-x[rows],inverse_metrics(lambda t:cubic_basis(t)@c,initial,rows),inverse_metrics(rational_fit(b,z,v,cfg),initial,rows)]
    metrics = np.array([[np.max(np.abs(e)),np.max(np.abs(e.sum(1)))] for e in errors])
    worst = np.maximum(worst,metrics)
    print(name, 'base/cubic/rational [channel,sum]:',metrics.tolist(),'best inner rational/base MSE:',float(inner_score.min()/base_score))
print('worst across slices, base/cubic/rational:',worst.tolist())
print('dataset keys:',list(ds),'calibration keys:',list(cal),'batch entries: 1; lab and stratum belong to trial records, not paired bridge rows')
