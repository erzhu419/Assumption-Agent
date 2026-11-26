import importlib.util, pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, _, y, complete = s._prepare(d)
complete = complete[:len(y)]
lab, strata = (d['train'][k] for k in ('lab_id', 'stratum'))
sy, t = d['yield_stress'], tr.times
labs = np.unique(lab)
full, ablations = np.empty_like(y), [np.empty_like(y), np.empty_like(y)]
correction = np.zeros(len(y))

def predict(use):
    lg, b = s._fit(tr, y, use & complete)
    return tr.matrix(lg) @ b

for c in labs:
    holdout = lab == c
    full[holdout] = predict(~holdout)[holdout]
    for j, name in enumerate(('over', 'work')):
        col = getattr(tr, name)
        setattr(tr, name, np.zeros_like(col))
        try:
            ablations[j][holdout] = predict(~holdout)[holdout]
        finally:
            setattr(tr, name, col)
    inner = np.empty_like(y)
    for v in labs[labs != c]:
        inner[lab == v] = predict((lab != c) & (lab != v))[lab == v]
    # Choose a rate update without using any outer-lab outcomes.
    for m, grid in ((True, [0., .25, .5, .75, 1.]), (False, [.5, .75, 1.])):
        use = ~holdout & (complete == m)
        e = y[use] - inner[use]
        cost = [np.mean(np.minimum(abs(e[:, :-1] - w*e[:, -1,None]*t), .10)) for w in grid]
        w = grid[np.argmin(cost)]
        correction[holdout & (complete == m)] = w
        print('inner-selected rate update, lab/complete/weight', c, m, w)

err = y - full
active_hist = np.any(np.abs(tr.past) > sy, axis=1)
for label, rows in [('subyield', ~active_hist & (tr.over == 0)), ('positive', tr.over > 0), ('negative', tr.over < 0), ('top-work', abs(tr.work) >= np.quantile(abs(tr.work), .9))]:
    def metrics(p):
        e = y[rows] - p[rows]
        return np.round([np.mean(abs(e[:, -1])), np.median(abs(e[:,:-1]))], 5)
    print('ablation', label, rows.sum(), 'full/without-hold/without-history rateMAE,curveMedAE', *(metrics(p) for p in [full,*ablations]))
for c in labs:
    for st in np.unique(strata):
        rows = (lab == c) & (strata == st) & complete
        print('cell', c, st, rows.sum(), 'median signed rate,last-horizon residual', np.round(np.median(err[rows][:,[-1,-2]],axis=0),5))
fixed = np.where(complete, .25, 1.)
for label, rows in [('all', np.ones(len(y),bool)), ('complete', complete), ('missing', ~complete), ('positive', tr.over > 0), ('negative', tr.over < 0), ('top-work',abs(tr.work)>=np.quantile(abs(tr.work),.9)), *[(f'lab{c}', lab==c) for c in labs], *[(f'stratum{j}',strata==j) for j in np.unique(strata)]]:
    def metrics(w):
        e = err[rows,:-1] - w[rows,None]*err[rows,-1,None]*t
        return np.round([np.mean(abs(e)), np.median(abs(e)), np.sqrt(np.mean(e**2))], 5)
    print('rate-update',label,'fixed/nested curve MAE,MedAE,RMSE',metrics(fixed),metrics(correction))

# Isolate the two activation paths at unchanged time constants and coupon features.
p = s._Design(np.array([[sy],[sy+.5],[sy],[sy+.5]]), np.array([sy,sy,sy+.25,sy+.25]), np.array([.4]), np.array([0.,.1,1.]), np.zeros((4,4)), sy)
a = p.matrix(np.log([.56,3.82]))
np.testing.assert_array_equal(np.any(a[..., -2:] != 0, axis=1), [[False,False],[False,True],[True,False],[True,True]])
print('independent activation verified')
