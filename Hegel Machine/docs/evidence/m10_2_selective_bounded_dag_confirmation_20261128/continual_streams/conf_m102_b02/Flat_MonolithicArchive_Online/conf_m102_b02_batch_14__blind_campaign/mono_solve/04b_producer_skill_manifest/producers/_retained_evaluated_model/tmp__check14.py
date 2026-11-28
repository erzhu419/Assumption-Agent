import numpy as np,pickle,importlib.util
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];n=len(tr['x']);times=s['eval_times']; scalar,temp=m._features(s,np.r_[tr['x'],s['test']['x']]);y=tr['y'];T=tr['traj'];coef=m._ls(temp[:n],T)
print('temporal coefficient time dependence',np.round(coef,5),sep='\n'); print('training RMSE',np.std(temp[:n]@coef-T));print('rank sv',np.linalg.svd(temp[:n]@coef,compute_uv=False))
for label,fun,start in [('two exp',lambda t,p:np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)],[1.8,.5]),('exp and t exp',lambda t,p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)],[1.8,.1]),('exp and t',lambda t,p:np.c_[-np.expm1(-p[0]*t),t],[1.8])]:
 def e(p):
  r=fun(times,p);return (r@m._ls(r,coef.T)-coef.T).ravel()
 q=least_squares(e,start).x; print('time model',label,q,np.std(e(q)))

def cv(te):
 trn=np.setdiff1d(np.arange(n),te); modes=np.linalg.svd(T[trn],full_matrices=False)[2][:2]; yp=scalar[te]@m._ls(scalar[trn],y[trn]);tp=temp[te]@m._ls(temp[trn],T[trn]@modes.T)@modes;return [abs(yp-y[te]).mean(),abs(tp-T[te]).mean(),abs(tp-T[te]).mean(0).max(),abs(yp-y[te]).max(),abs(tp-T[te]).max()]
print('prescribed',cv(np.arange(52,72))); rng=np.random.default_rng(14); print('random 52/20 worst:',np.max([cv(rng.choice(n,20,False)) for _ in range(100)],0));
for j in range(4):
 ix=np.argsort(tr['x'][:,j]); print('coordinate',j,cv(ix[:20]),cv(ix[-20:]))
