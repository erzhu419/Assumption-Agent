import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']
a = up._design(x); rows = np.arange(len(x)); rng = np.random.default_rng(180)
full = up._fit_forward(x,y)
clean = np.linalg.norm(a@full-y,axis=1)<1
folds = {'axis':rows[np.count_nonzero(x,axis=1)==1], 'outer':np.argsort(abs(x).max(axis=1))[-12:], 'high total':np.argsort(x.sum(axis=1))[-12:], 'low total':np.argsort(x.sum(axis=1))[:12]}
for k in ('train','test','chamber_verification_trials'):
    if d[k] is not None:
        for field in ('past_channel_raw','hold_channel_raw'):
            r = d[k][field]
            print(k,field,'samples',r.size//4)
operating = np.concatenate([d[k][f].reshape(-1,4) for k in ('train','test','chamber_verification_trials') if d[k] is not None for f in ('past_channel_raw','hold_channel_raw')]); ref = up._invert(operating,full)
# Exact rational challengers, including pairwise sum/difference denominators.
# Denominator selection uses only inner folds of the outer training rows.
for name, te in folds.items():
    tr = np.setdiff1d(rows,te); te = te[clean[te]]
    fit = up._fit_forward(x[tr],y[tr]); good = tr[np.linalg.norm(a[tr]@fit-y[tr],axis=1)<1]
    inner = np.array_split(rng.permutation(good),3)
    base = a[te]@fit; rational = base.copy(); sparse = base.copy(); selected = []
    for j in range(4):
        q = x[:,j]; other = x[:,(j+1)%4]
        denoms = [np.ones(len(x))] + [1+s*z*z for z in (q,q+other,q-other) for s in (0.02,0.10,0.30)]
        test_losses=[]; preds=[]
        for den in denoms:
            m=a/den[:,None]; loss=[]
            for val in inner:
                sub=np.setdiff1d(good,val); co=np.linalg.lstsq(m[sub],y[sub,j],rcond=None)[0]
                loss.extend((m[val]@co-y[val,j])**2)
            test_losses.append(np.mean(loss)); preds.append(m[te]@np.linalg.lstsq(m[good],y[good,j],rcond=None)[0])
        best=int(np.argmin(test_losses)); rational[:,j]=preds[best]; selected.append(best)
        use = abs(fit[:,j]) > .001
        sparse[:,j] = a[te][:,use] @ np.linalg.lstsq(a[good][:,use],y[good,j],rcond=None)[0]
    inv=up._invert(y[te],fit)-x[te]; drift=np.max(abs(up._invert(operating,fit)-ref))
    print(name, 'n',len(te),'forward RMS current/rational/sparse',*[float(np.sqrt(np.mean((p-y[te])**2))) for p in (base,rational,sparse)], 'rational selections',selected,'inverse peak',np.max(abs(inv)),'operating refit drift',drift)
    assert np.max(abs(inv))<.002 and drift<.002
out=solve_batch(d)
assert np.shape(out['predicted_strain'])==(len(d['test']['lab_id']),len(d['eval_times'])) and all(np.all(np.isfinite(out[k])) for k in ('predicted_strain','initial_rate_pred','hereditary_weights','estimated_taus'))
print('isolated composition passed')
