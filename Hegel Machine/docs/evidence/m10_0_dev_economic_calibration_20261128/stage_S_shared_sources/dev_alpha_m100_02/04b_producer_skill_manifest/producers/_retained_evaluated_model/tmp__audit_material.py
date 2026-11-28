import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
x, _ = m._features(tr, te)
s = m._prepare(tr, x, m._bridge(d['calibration_load_cells']))
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]
r = tr['instant_rate_obs']
load = s[1]
history = load - s[0] @ (dt / dt.sum())
lab = tr['lab_id']
lab_terms = (lab[:,None] == np.unique(lab)).astype(float)
aux = [np.c_[v, v*load, v*history] for v in x.T]
extras = {
    'lab-offset': lab_terms,
    'pairwise': np.column_stack(aux + [load*history, (load+history)**2, (load-history)**2]),
    'rational': np.column_stack([v/(1+q*q) for v in [load, history, *x.T] for q in (load, history)]),
}
for name, group in [('lab',lab), ('stratum',tr['stratum']), ('load',np.digitize(load,np.quantile(load,[1/3,2/3])))]:
    for g in np.unique(group):
        h = group == g
        f = ~h
        tau,w = m._fit(tuple(v[f] for v in s), dt,t,y[f],r[f])
        C,R = m._design(np.log(tau),s,dt,t)
        e, q = y-C@w, r-R@w
        ah = np.abs(e[h])
        j = np.argmax(np.quantile(ah,.9,axis=0))
        print(name,g,'horizon',round(float(t[j]),3),'signed med',round(float(np.median(e[h,j])),4),'p90',round(float(np.quantile(ah[:,j],.9)),4),'count >0.1',int((ah[:,j]>.1).sum()))
        print('  median abs strain,rate:',*(round(float(v),5) for v in [np.median(ah),np.median(np.abs(q[h]))]),end='')
        for label,A in extras.items():
            A = A-A[f].mean(0)
            A /= np.maximum(A[f].std(0),1e-7)
            A[:,A[f].std(0)<1e-6] = 0
            sf,sr=max(m._mad(e[f]),1e-7),max(m._mad(q[f]),1e-7)
            B=(A[f,None,:]*t[None,:,None]).reshape(-1,A.shape[1])/sf
            Z=np.vstack([B,A[f]/sr,np.eye(A.shape[1])*50])
            target=np.r_[e[f].ravel()/sf,q[f]/sr,np.zeros(A.shape[1])]
            w1=least_squares(lambda w: Z@w-target,np.zeros(A.shape[1]),jac=lambda w: Z.copy(),loss='soft_l1',max_nfev=50).x
            corr=A[h]@w1
            changed=np.median(np.abs(e[h]-corr[:,None]*t)),np.median(np.abs(q[h]-corr))
            print(' ',label,*(round(float(v),5) for v in changed),end='')
        print()
