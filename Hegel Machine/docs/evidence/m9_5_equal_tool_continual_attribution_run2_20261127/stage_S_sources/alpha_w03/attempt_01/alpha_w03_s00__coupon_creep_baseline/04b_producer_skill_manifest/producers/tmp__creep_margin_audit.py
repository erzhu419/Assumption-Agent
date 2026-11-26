import importlib.util
import itertools
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_w03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_w03_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
a = d['train']; n = len(a['lab_id']); lab = a['lab_id']; wing = lab//2
x,h,p = (v[:n] for v in m._prep(d)); dt,t = d['past_dt'],d['eval_times']
y = a['strain_curves_obs'] - a['strain_0'][:,None]; r = a['instant_rate_obs']
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)
q = x[:,1:]; pairs = list(itertools.combinations(range(4),2))
extra = np.column_stack([q**2, q/(1+q**2)] + [(q[:,i]*q[:,j]) for i,j in pairs] + [((q[:,i]-q[:,j])/(1+(q[:,i]+q[:,j])**2)) for i,j in pairs] + [np.prod(q[:,ijk],axis=1) for ijk in itertools.combinations(range(4),3)])

def mad(z):
    return max(1e-6,1.4826*np.median(np.abs(z-np.median(z))))

def metrics(ec,er):
    return np.array([np.median(abs(ec)),np.quantile(abs(er),.9),np.max(np.median(abs(ec),axis=0))])

for name,group in [('wing',wing),('lab',lab),('stratum',a['stratum'])]:
    summary=[]
    for g in np.unique(group):
        v = group == g; f = ~v
        tau,w = m._fit(x[f],h[f],p[f],dt,t,y[f],r[f]); c,b,*_ = m._design(tau,x,h,p,dt,t)
        ec=c@w-y; er=b@w-r
        cf=ec[f]; rf=er[f]; sc=2.8*mad(cf); sr=2.8*mad(rf)
        # Fit only on the outer training fold. The augmented terms are extra linear-in-time drift.
        z=extra[f]; design=np.concatenate(((z[:,None,:]*t[None,:,None]).reshape(-1,z.shape[1])/sc,z/sr))
        residual=np.r_[cf.ravel()/sc,rf/sr]
        aug=least_squares(lambda u: residual+design@u,np.zeros(z.shape[1]),jac=lambda u: design.copy(),loss='cauchy',max_nfev=100).x
        delta=extra[v]@aug
        first=metrics(ec[v],er[v]); second=metrics(ec[v]+delta[:,None]*t,er[v]+delta)
        summary.append(np.r_[first,second])
        clean=v&complete
        print(name,int(g),'base',first.round(6),'nonlinear drift',second.round(6),'complete signed median curve/rate',np.round([np.median(ec[clean]),np.median(er[clean])],6),'taus',tau.round(5))
    print(name,'worst base/augmented',np.max(summary,axis=0).round(6))

out=m.solve_alpha_w03_s00(d); tau=out['estimated_taus']; w=out['hereditary_weights']
c,b,*_=m._design(tau,x,h,p,dt,t)
step=1e-6
cstep,*_=m._design(tau,x,h,p,dt,np.array([step]))
err=np.max(np.abs(cstep[:,0]@w/step-b@w)); assert err<1e-5,err
assert np.array_equal(out['viscoplastic_weights'],np.zeros(2))
print('tau box/order margins',np.array([tau[0]-.2,tau[1]-tau[0],8-tau[1]]),'rate finite-difference max error',err,'design rank',np.linalg.matrix_rank(np.vstack((c.reshape(-1,7),b))))
print('nuisance metadata keys',list(d),'; dataset has continuous outcomes, no class labels')
