import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('batch','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s); Z=s['Z_nuisance']; a=s['aux_manifold_telemetry']; print('aux mean/std',a.mean(0).round(4),a.std(0).round(4))
allidx=np.arange(len(y)); folds=np.array_split(allidx,len(s['mu_shifts']))
for name,laws in [('closed',((0,0),(0,0))),('closed_y_periodic',((0,0),'periodic')),('robin',((1,1),(1,1)))]:
 S,_=m._forward(s,laws); out=[]
 for ridge in [0.,s['source_ridge']]:
  errs=[]
  for idx in [allidx]+folds:
   tr=allidx if len(idx)==len(y) else np.setdiff1d(allidx,idx)
   A=m._project(Z[tr],s['nuisance_ridge'],np.c_[S[tr],y[tr]]); w,nu=m._qp(A[:,:-1],A[:,-1],ridge,s['u_box'],s['total_budget']); gamma=np.linalg.solve(Z[tr].T@Z[tr]+s['nuisance_ridge']*np.eye(Z.shape[1]),Z[tr].T@(y[tr]-S[tr]@w)); errs.append(np.sqrt(np.mean((y[idx]-S[idx]@w-Z[idx]@gamma)**2)))
  out.append(np.round(errs,7))
 print(name,'full and held-out shift RMSE at zero/requested ridge',*out)
# Unpenalized fit isolates systematic wall errors from source and nuisance shrinkage.
Q=np.linalg.qr(Z,mode='reduced')[0]; P=lambda x:x-Q@(Q.T@x); yp=P(y)
def fit(p,ret=False):
 S,_=m._forward(s,[p[:2],p[2:]]); S=P(S); A=S[:,:-1]-S[:,[-1]]; v=np.linalg.lstsq(A,yp-s['total_budget']*S[:,-1],rcond=None)[0]; w=np.r_[v,s['total_budget']-v.sum()]; e=S@w-yp; return (w,np.sqrt(np.mean(e**2))) if ret else e
for p in (np.zeros(4),least_squares(fit,np.zeros(4),bounds=(-.2,1.5),gtol=1e-12).x): print('Robin scales',p,'OLS weights/RMSE',fit(p,True))
out=m.solve_batch(s); w=out['weights']; c=out['kkt_certificate']; print('budget error',abs(w.sum()-s['total_budget']),'KKT gap',c['primal_objective']-c['dual_lower_bound'])
