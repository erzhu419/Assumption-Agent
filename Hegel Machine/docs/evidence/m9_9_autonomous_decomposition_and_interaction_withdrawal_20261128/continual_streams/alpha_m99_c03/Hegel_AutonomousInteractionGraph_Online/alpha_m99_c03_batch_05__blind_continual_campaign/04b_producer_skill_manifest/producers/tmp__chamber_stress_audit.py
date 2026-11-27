import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _chamber, _coupon_inputs, _design, _features, _fit
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
cal = calibrate_upstream({**raw, 'chamber_verification_trials': item['campaign_diagnostics']})
tr, te, q = (cal[k] for k in ('train','test','chamber_verification_trials'))
t, dt = (np.asarray(cal[k]) for k in ('eval_times','past_dt'))
out = solve_batch(raw)
a,w = out['estimated_taus'],out['hereditary_weights']
assert 0.05 < a[0] < a[1] < 25
x = _features(tr,te)
n = len(tr['lab_id'])
_, hte, fte = _coupon_inputs(te,x[n:])
for key,shape in {'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (len(hte),len(t)), 'initial_rate_pred': (len(hte),)}.items():
    assert out[key].shape == shape and np.isfinite(out[key]).all(), key
p0,r0 = _chamber(a,w,hte,fte,np.array([0.,1e-7]))
np.testing.assert_allclose(p0[:,0],0,atol=1e-12)
np.testing.assert_allclose(p0[:,1]/1e-7,out['initial_rate_pred'],rtol=2e-7,atol=1e-7)
lab = np.asarray(q['lab_id']); ref = q['ref_coupons']; xx = q['coupon_features_raw'].astype(float).copy()
complete = np.isfinite(xx).all(1)
for g in np.unique(lab):
    k = lab == g
    xx[k] = (xx[k]-np.median(ref[k].mean(0)))/np.median(ref[k].std(0))
xx = np.where(np.isfinite(xx), xx, np.nanmean(xx,axis=0))
inputs = _coupon_inputs(q,xx)
past,h,fq = inputs
tr_in = _coupon_inputs(tr,x[:n]); yt = tr['strain_curves_obs']-tr['strain_0'][:,None]; rt = tr['instant_rate_obs']
y = np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]
fixed = np.column_stack(_chamber(a,w,h,fq,t))
names = ('bench transfer','archive transfer','reset mixture','preload contrast','load-pair drift interaction','rational load drift','lab intercept')
pred = {k:np.empty_like(y) for k in names}
for g in np.unique(lab):
    k = lab == g; fit = ~k & complete
    bench_fit = tr['lab_id'] != g
    af,wf = _fit(tuple(z[bench_fit] for z in tr_in),dt,t,yt[bench_fit],rt[bench_fit])
    E = -np.expm1(-t[:,None]/af)
    base = h[:,None]*np.r_[E@wf[:2],np.sum(wf[:2]/af)]
    dr = fq@wf[2:]
    J = np.vstack((E*af,np.ones(2)))
    D = np.column_stack((np.r_[t,1.],J))
    Z = D[:,:2]-D[:,2:]
    M = dr[:,None,None]*Z
    target = y-base-dr[:,None]*D[:,2]
    coef = np.linalg.lstsq(M[fit].reshape(-1,2),target[fit].ravel(),rcond=None)[0]
    ker = D@np.r_[coef,1-coef.sum()]
    pred['reset mixture'][k] = base[k]+dr[k,None]*ker
    pred['archive transfer'][k] = base[k]+dr[k,None]*J.mean(1)
    ac,ar = _design(af,inputs,dt,t)
    pred['bench transfer'][k] = np.c_[ac@wf,ar@wf][k]
    memory = h[:,None]-ar[:,:2]*af
    H = np.asarray(q['hold_channel_raw'])
    pairs = np.column_stack([H[:,i]-H[:,j] for i in range(4) for j in range(i+1,4)])
    hcenter = h-np.median(h[fit]); qscale = np.std(hcenter[fit])
    v = hcenter/qscale
    rq = dr/(1+v*v)-dr
    labcols = (lab[:,None] == np.unique(lab)[None,:]).astype(float)
    extras = {'preload contrast': memory[:,:,None]*(J/af).T[None,:,:],
              'load-pair drift interaction': (pairs*dr[:,None])[:,:,None]*ker,
              'rational load drift': rq[:,None,None]*ker,
              'lab intercept': labcols[:,:,None]*ker}
    for name,A in extras.items():
        B = np.concatenate((M,A.transpose(0,2,1)),axis=2)
        c = np.linalg.lstsq(B[fit].reshape(-1,B.shape[2]),target[fit].ravel(),rcond=1e-10)[0]
        pred[name][k] = (base+dr[:,None]*D[:,2]+B@c)[k]

def rmse(e):
    return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
for name,p in pred.items():
    e = p-y
    print(name, 'pooled',np.round(rmse(e),6),'worst held lab',np.round(np.max([rmse(e[lab==g]) for g in np.unique(lab)],axis=0),6))
e = fixed-y
print('exact artifact by lab:', [(int(g),rmse(e[lab==g]).round(6).tolist()) for g in np.unique(lab)])
print('exact artifact complete/missing, high absolute load, late strain:',rmse(e[complete]),rmse(e[~complete]),rmse(e[np.abs(h)>=np.quantile(np.abs(h),.75)]),np.sqrt(np.mean(e[:,8:-1]**2)))
assert np.all(np.max([rmse(e[lab==g]) for g in np.unique(lab)],axis=0)<.012)
