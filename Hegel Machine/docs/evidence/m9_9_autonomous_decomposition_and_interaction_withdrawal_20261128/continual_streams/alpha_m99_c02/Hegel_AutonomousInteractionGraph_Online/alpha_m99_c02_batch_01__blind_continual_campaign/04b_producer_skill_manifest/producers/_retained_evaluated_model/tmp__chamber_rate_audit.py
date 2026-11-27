import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];a=i['campaign_diagnostics'];tr,te=d['train'],d['test'];t=d['eval_times'];out=s.solve_batch(d);ts,w=out['estimated_taus'],out['hereditary_weights'];b=s._bridge(d['calibration_load_cells']);p,h=s._loads(a,b);_,x=s._features({k:v[:0] for k,v in a.items()},a);r=w[2]+x@w[3:];g=s._basis(ts,p,h,x,d['past_dt'],t)[1][:,:2]*ts;P=np.vstack((-np.expm1(-t[:,None]/ts),1/ts));base=(h[:,None]*w[:2])@P.T;Y=np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']];L=a['lab_id']; c=d['calibration_load_cells'];raw=c['bridge_raw'];load=c['true_channel_loads'].sum(1);lin=np.c_[np.ones(len(raw)),raw];quad=np.c_[lin,*[raw[:,j]*raw[:,k] for j in range(4) for k in range(j,4)]]
A=r[:,None,None]*P[None,:,:]; F=np.concatenate((A,r[:,None,None]*np.r_[t,1][None,:,None]),axis=2)
for name,M in [('two modes',A),('two modes + steady drift',F)]:
 E=np.empty_like(Y)
 for lab in np.unique(L):
  m=L==lab; q=np.linalg.lstsq(M[~m].reshape(-1,M.shape[-1]),(Y-base)[~m].ravel(),rcond=None)[0]; E[m]=(base+M@q-Y)[m]
  print(name,'held-out lab',lab,'parameters',q.round(6),'strain/rate/worst-horizon RMSE',np.sqrt(np.mean(E[m,:-1]**2)),np.sqrt(np.mean(E[m,-1]**2)),np.sqrt(np.mean(E[m]**2,axis=0)).max())
 q=np.linalg.lstsq(M.reshape(-1,M.shape[-1]),(Y-base).ravel(),rcond=None)[0]; print(name,'all-data coefficients',q,'implied drift contribution to initial rate',q[:2]@(1/ts)+(q[2] if len(q)>2 else 0))
 for name,m in [('missing',np.isnan(a['coupon_features_raw']).any(1)),('large absolute drift',abs(r)>=np.median(abs(r)))]: print(name,'curve/rate RMSE',np.sqrt(np.mean(E[m,:-1]**2)),np.sqrt(np.mean(E[m,-1]**2)))
