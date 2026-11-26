import importlib.util
import itertools
import pickle
import numpy as np
from scipy.optimize import least_squares

path = '/holdout_task/alpha_ep04_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
lab = d['train']['lab_id']
strata = d['train']['stratum']
complete = np.isfinite(d['train']['coupon_features_raw']).all(axis=1)
labels = ('base', 'stress_drift', 'pairwise_drift', 'rational_memory', 'lab_level')
pred = {k: np.empty((len(lab), len(d['eval_times']) + 1)) for k in labels}
for l in np.unique(lab):
    m = lab == l
    train = {k:v[~m] for k,v in d['train'].items()}
    test = {k:v[m] for k,v in d['train'].items() if k != 'strain_curves_obs'}
    test['residual_plastic_state'] = np.zeros(m.sum())
    q = {**d, 'train':train, 'test':test}
    tr, te, y, seen = s._prepare(q)
    eligible = seen[:len(y)].all(axis=1)
    tau, w = s._fit(tr, y, eligible)
    A, B = (z.matrix(np.log(tau)) for z in (tr,te))
    residual = y[eligible] - A[eligible] @ w
    scale = s._scale(residual)
    def extra(z, p, a):
        t = np.r_[z.times, 1.]
        x = z.linear[:, -1, 1:]
        ij = list(itertools.combinations_with_replacement(range(x.shape[1]), 2))
        pairwise = np.column_stack([x[:,i]*x[:,j] for i,j in ij])
        return dict(stress_drift=z.hold[:,None,None]*t[None,:,None],
                    pairwise_drift=pairwise[:,None,:]*t[None,:,None],
                    rational_memory=a[:,:,:2]/(1+0.25*z.hold[:,None,None]**2),
                    lab_level=(p['lab_id'][:,None,None]==np.unique(train['lab_id'])[None,None,1:])*t[None,:,None])
    ea, eb = extra(tr,train,A), extra(te,test,B)
    for name in labels:
        if name == 'base':
            p = B @ w
        else:
            a = np.concatenate([A,ea[name]],axis=2)[eligible]
            b = np.concatenate([B,eb[name]],axis=2)
            aw = (a/scale[None,:,None]).reshape(-1,a.shape[-1]); yw=(y[eligible]/scale).ravel()
            fit = least_squares(lambda v: aw@v-yw, np.r_[w,np.zeros(a.shape[-1]-len(w))], jac=lambda v: aw.copy(), loss='cauchy', f_scale=2.)
            p = b @ fit.x
        rate = test['instant_rate_obs']
        adj = np.where(~complete[m] & np.isfinite(rate), rate-p[:,-1], 0.)
        pred[name][m] = p + adj[:,None]*np.r_[te.times,1.]
obs = d['train']; curves = obs['strain_curves_obs']-obs['strain_0'][:,None]
for name,p in pred.items():
    err = p[:,:-1]-curves
    print(name)
    for cm in (complete, ~complete):
        cells = [cm&(lab==l)&(strata==k) for l in np.unique(lab) for k in np.unique(strata)]
        print('  n',cm.sum(),'curve MAE / worst cell n>=3 / median:',*[round(v,6) for v in (np.mean(abs(err[cm])),max(np.mean(abs(err[c])) for c in cells if c.sum()>=3),np.median(abs(err[cm])))], 'lab signed medians:',np.round([np.median(err[cm&(lab==l)]) for l in np.unique(lab)],6))
    print('  complete rate MAE / worst lab:',*[round(v,6) for v in (np.mean(abs(p[complete,-1]-obs['instant_rate_obs'][complete])), max(np.mean(abs(p[complete&(lab==l),-1]-obs['instant_rate_obs'][complete&(lab==l)])) for l in np.unique(lab)))])
print('test finite rates by completeness',[(bool(v), int(np.sum(m)),int(np.sum(m & np.isfinite(d['test']['instant_rate_obs'])))) for v in (True,False) for m in (np.isfinite(d['test']['coupon_features_raw']).all(axis=1)==v,)])
