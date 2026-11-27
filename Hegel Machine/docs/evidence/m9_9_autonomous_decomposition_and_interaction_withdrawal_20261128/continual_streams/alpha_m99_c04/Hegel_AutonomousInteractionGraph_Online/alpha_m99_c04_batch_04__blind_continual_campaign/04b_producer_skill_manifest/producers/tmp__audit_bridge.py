import json, pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import upstream_transducer as m
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
with open('/tmp/hegel_continual_archive/archive_apparatus_log.json') as f:
    log = json.load(f)
print('Apparatus log (prefix):', json.dumps(log)[-3800:])
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
a = m._design(x)
nxt, prv = x[:, m._NEXT], x[:, m._PREV]
# A squared difference adds a missing neighboring quadratic; saturation tests
# the corresponding rational basis. Neither changes the base invertible model.
def add_features(kind):
    q = x - nxt
    if kind == 'difference':
        return (q*q)[..., None]
    if kind == 'rational':
        return np.stack([q**2/(1+s*q**2) for s in (.15, .6)] + [x**3/(1+.3*q**2)], axis=-1)
    return np.empty((*x.shape, 0))
def pred(z, b):
    return np.einsum('nif,if->ni', z, b)
axis = np.count_nonzero(x, axis=1) == 1
mix = np.flatnonzero(~axis)
folds = [(f'axis{i}', np.flatnonzero(axis & (x[:, i] != 0))) for i in range(4)]
folds += [('high mixed', mix[np.argsort(np.max(abs(x[mix]), axis=1))[-max(8, len(mix)//5):]])]
folds += [(f'mixed{i}', v) for i, v in enumerate(np.array_split(np.random.default_rng(9).permutation(mix), 5))]
worst = {}
for label, val in folds:
    tr = ~np.isin(np.arange(len(x)), val)
    b = m._fit_bridge(x[tr], y[tr]); train_r = np.linalg.norm(pred(a[tr], b)-y[tr], axis=1)
    clean_tr = np.flatnonzero(tr)[train_r < .01]
    clean_val = val[np.linalg.norm(pred(a[val], b)-y[val], axis=1) < .01]
    if not len(clean_val):
        continue
    # Test whether the four channels are scaled copies of the same transfer law.
    def unpack(p):
        return np.c_[p[:4], p[4:8, None]*np.r_[1.,p[8:]][None, :]]
    p0 = np.r_[b[:,0], b[:,1], np.median(b[:,2:]/b[:,1,None], axis=0)]
    shared = unpack(least_squares(lambda p: (pred(a[clean_tr], unpack(p))-y[clean_tr]).ravel(), p0).x)
    z0 = x[clean_val]; obs = y[clean_val]
    for kind in ('base', 'shared', 'difference', 'rational'):
        z = np.concatenate([a, add_features(kind)], axis=-1)
        coef = (b if kind == 'base' else shared if kind == 'shared' else np.stack([np.linalg.lstsq(z[clean_tr, j], y[clean_tr,j], rcond=None)[0] for j in range(4)]))
        err = np.max(np.abs(pred(z[clean_val], coef)-obs))
        worst[kind] = max(worst.get(kind,0.), err)
        print(label, kind, 'forward_max', format(err,'.7g'), end='; ')
        if kind in ('base','shared'):
            e = m._inverse(obs,coef)-z0
            print('inverse_chan/sum', format(np.max(abs(e)),'.7g'),format(np.max(abs(e.sum(axis=1))),'.7g'),end='; ')
    print('clean/rejected', len(clean_val), len(val)-len(clean_val))
print('Worst forward per-channel error:', worst)
for part in ('train','test'):
    print(part, 'laboratories (counts):', np.unique(d[part]['lab_id'], return_counts=True), 'strata:', np.unique(d[part]['stratum'], return_counts=True))
print('Bridge calibration fields:', sorted(c), 'classification labels or relay modes: none in supplied calibration schema')
