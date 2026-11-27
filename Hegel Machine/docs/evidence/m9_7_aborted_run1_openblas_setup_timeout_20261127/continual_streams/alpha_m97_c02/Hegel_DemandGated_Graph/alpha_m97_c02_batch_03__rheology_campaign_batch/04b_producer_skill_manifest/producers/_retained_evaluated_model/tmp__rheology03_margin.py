import importlib.util, pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']; n = len(tr['lab_id'])
past, hold, f, e0 = m._prepare(d)
lab, st, raw = (np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw'))
missing = ~np.isfinite(raw)
observed = np.where(missing, np.nan, f[:,1:])
cell = np.array([[np.nanmean(observed[(lab==l)&(st==g)],axis=0) for g in range(3)] for l in range(4)])
variants = {'global_cell_balanced': f[:,1:], 'local_cell': cell[lab,st], 'stratum_balanced': np.nanmean(cell,axis=0)[st], 'lab_balanced': np.nanmean(cell,axis=1)[lab]}
y, r = tr['strain_curves_obs']-e0[:n,None], tr['instant_rate_obs']; dt, t = d['past_dt'], d['eval_times']
for key, fill in variants.items():
    feat = np.column_stack((f[:,0], np.where(missing,fill,observed)))[:n]
    ce, re = np.empty_like(y), np.empty_like(r)
    for l in range(4):
        v = lab[:n]==l; a = ~v
        tau,w=m._fit(past[:n][a],hold[:n][a],feat[a],dt,t,y[a],r[a])
        cb,rb=m._basis(tau,past[:n][v],hold[:n][v],feat[v],dt,t)
        ce[v],re[v]=cb@w-y[v],rb@w-r[v]
    ms=missing[:n].any(axis=1)
    def metrics(v): return round(np.median(abs(ce[v])),6),round(np.sqrt(np.mean(re[v]**2)),6)
    print(key, 'missing n=',ms.sum(),'missing OOF',metrics(ms),'complete OOF',metrics(~ms),'worst missing-lab rate RMSE',max(metrics(ms&(lab[:n]==l))[1] for l in np.unique(lab[:n][ms])))
    if key=='global_cell_balanced':
        for l in range(4):
            for g in range(3):
                v=(lab[:n]==l)&(st[:n]==g)
                print('OOF lab/stratum',l,g,'rate median bias',round(np.median(re[v]),6),'count',v.sum())
# Verify each basis column separately, so a cancellation cannot hide a missing term.
tau,w=m._fit(past[:n],hold[:n],f[:n],dt,t,y,r)
z=np.zeros((len(hold),2))
for j,h in enumerate(dt):
    q=np.exp(-h/tau); z=z*q+past[:,j,None]*(1-q)
gap=hold[:,None]-z
b,br=m._basis(tau,past,hold,f,dt,t)
u0=1-np.exp(-t/tau[0]); u1=1-np.exp(-t/tau[1]); phi=t/(1+.85*t)
slip=np.sign(hold)*np.sqrt(abs(hold)); delta=gap[:,0]-gap[:,1]
expected=np.stack([gap[:,0,None]*u0,gap[:,1,None]*u1,*[f[:,i,None]*phi for i in range(5)],slip[:,None]*phi,delta[:,None]*u0],axis=-1)
np.testing.assert_allclose(b,expected,rtol=0,atol=1e-12)
np.testing.assert_allclose(br,np.column_stack((gap/tau,f,slip,delta/tau[0])),rtol=0,atol=1e-12)
print('Independent term-by-term constitutive audit passed')
