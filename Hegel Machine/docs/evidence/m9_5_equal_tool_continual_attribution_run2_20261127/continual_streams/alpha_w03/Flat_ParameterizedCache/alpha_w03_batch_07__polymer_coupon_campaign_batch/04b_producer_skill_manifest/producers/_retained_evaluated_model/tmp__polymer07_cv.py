import importlib.util, pickle
import numpy as np
root = '/holdout_task/'
s = importlib.util.spec_from_file_location('candidate',root+'alpha_w03_batch_07_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_w03_batch_07.pkl','rb'))[0]['dataset']
a = d['train']; lab = a['lab_id']; n = len(lab)
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)
z = a['strain_curves_obs']-a['strain_0'][:,None]; rate = a['instant_rate_obs']
x,h,p = (v[:n] for v in m._prep(d)); dt,t,q = (d[k] for k in ('past_dt','eval_times','yield_stress'))
rng = np.random.default_rng(714); fold = np.empty(n,int)
for j in range(4):
    for c in (False,True):
        ids = rng.permutation(np.flatnonzero((lab==j)&(complete==c)))
        fold[ids] = np.arange(len(ids)) % 4
err = np.empty((3,n,len(t))); rate_err = np.empty(n)
for k in range(4):
    tr, te = fold!=k, fold==k; fit = tr & complete
    tau,w = m._fit(x[fit],h[fit],p[fit],dt,t,q,z[fit],rate[fit])
    curves, rates = m._design(tau,x[te],h[te],p[te],dt,t,q)
    raw = curves@w-z[te]; innovation = rate[te]-rates@w; rate_err[te] = innovation
    wing = m._wing_blend(x[tr],h[tr],p[tr],dt,t,q,z[tr],rate[tr],lab[tr],complete[tr]); blend = np.where(complete[te],wing[lab[te]//2],1.)
    err[:,te] = [raw, raw+blend[:,None]*innovation[:,None]*t, raw+innovation[:,None]*t]
    print('fold',k,'taus',np.round(tau,5),'plastic',w[7:].tolist(),'drift blends',np.round(wing,4))
for name,sel in [('all', np.ones(n,bool)), ('imputed',~complete), ('complete',complete)]+[(f'lab {j}',lab==j) for j in range(4)]:
    aerr = err[:,sel]; print(name,'n',sel.sum(),'curve RMSE raw/current/full',np.round(np.sqrt(np.mean(aerr**2,axis=(1,2))),6),'late RMSE',np.round(np.sqrt(np.mean(aerr[:,:,-1]**2,axis=1)),6),'raw rate RMSE',round(np.sqrt(np.mean(rate_err[sel]**2)),6))
# An independent stepwise state propagation checks segmented history and plastic activation.
tau = np.array([.46,3.2]); time = np.r_[0.,1e-6,t]; dur = np.array([.2,.65,1.3]); hold = np.array([1.,3.8,-3.9,1.2]); past = np.array([[.3,-.5,1.],[.3,-.5,1.],[3.6,-3.8,1.],[-3.6,.5,1.]])
feat = np.column_stack([np.ones(4),np.arange(16).reshape(4,4)/10]); coeff = np.array([.7,.55,.08,.15,-.13,.11,-.09,.21,.17]); y0=np.zeros((4,2))
for j, delta in enumerate(dur):
    y0 = past[:,j,None]+(y0-past[:,j,None])*np.exp(-delta/tau)
work = np.sum(np.clip(abs(past)-q,0,None)**2*dur,axis=1)*hold
slope = feat@coeff[2:7]+coeff[7]*np.sign(hold)*np.clip(abs(hold)-q,0,None)**2
expected = (hold[:,None]-y0) @ (coeff[:2,None]*(1-np.exp(-time/tau[:,None]))) + slope[:,None]*time + coeff[8]*work[:,None]*(1-np.exp(-time/tau[0]))
C,R = m._design(tau,feat,hold,past,dur,time,q); diff = np.max(abs(C@coeff-expected)); dr = np.max(abs(R@coeff-expected[:,1]/time[1])); print('history/threshold max absolute error',diff,'onset finite-difference error',dr)
assert diff < 1e-12 and dr < 1e-5
