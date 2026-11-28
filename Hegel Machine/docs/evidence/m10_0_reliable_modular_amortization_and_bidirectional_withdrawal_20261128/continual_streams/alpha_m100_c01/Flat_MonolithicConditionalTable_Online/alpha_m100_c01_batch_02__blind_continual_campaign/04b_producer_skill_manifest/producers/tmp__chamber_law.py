import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('prev','/tmp/hegel_continual_archive/alpha_m100_c01_batch_01_solver.py'); prev=importlib.util.module_from_spec(s);s.loader.exec_module(prev)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; te=d['test']; g=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; ca=d['calibration_load_cells']; b=prev._bridge(ca['bridge_raw'],ca['true_channel_loads']); xt,xe=prev._features(tr,te)
P,H,F=prev._inputs(tr,xt,b); a,w=prev._fit(P,H,F,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',a,w,flush=True)
from scipy.optimize import minimize_scalar
x=prev._features(g,{k:v[:0] for k,v in g.items()})[0]
P,H,F=prev._inputs(g,x,b); C,R=prev._design(a,P,H,F,dt,t); Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; base=np.c_[C@w,R@w]; U=F@w[2:]; E=np.vstack((np.c_[-np.expm1(-t[:,None]/a),t],np.r_[1/a,1.])); S=Y-base; v=U@S/(U@U); print('drift time response',np.round(v,6),'final raw residual:',np.sqrt(np.mean(S*S,axis=0))[-2:]); print('lab/strata/missing',np.c_[g['lab_id'],g['stratum'],np.isnan(g['coupon_features_raw']).sum(axis=1)].tolist())
def val(name, cols):
 X=np.stack(cols,axis=-1); ii=np.arange(len(U)); pred=np.empty_like(Y)
 def fit(mask): return np.linalg.lstsq(X[mask].reshape(-1,X.shape[-1]),S[mask].ravel(),rcond=None)[0]
 for l in np.unique(g['lab_id']):
  mask=g['lab_id']==l; pred[mask]=X[mask]@fit(~mask)
 err=pred-S; rm=np.sqrt(np.mean(err**2,axis=0)); print(name,fit(ii>=0).round(6),'LOLO max time/rate',rm[:-1].max().round(6),rm[-1].round(6),'max lab rate',max(np.sqrt(np.mean(err[g['lab_id']==l,-1]**2)) for l in np.unique(g['lab_id'])).round(6))
val('score x free two branches + ramp',[U[:,None]*e for e in E.T]); val('score x two branches plus train drift removed',[U[:,None]*e for e in E.T[:2]]+[U[:,None]*E[:,2]]); val('score x zero-rate difference',[U[:,None]*(E[:,j]*a[j]-E[:,2]) for j in range(2)]); val('only slow branch, same initial rate',[U[:,None]*(E[:,1]*a[1]-E[:,2])]); val('only fast branch, same initial rate',[U[:,None]*(E[:,0]*a[0]-E[:,2])]);
def sat(tau): return np.r_[-tau*np.expm1(-t/tau)-t,0.]
q=minimize_scalar(lambda z: np.sum((S-U[:,None]*sat(z))**2),bounds=(.1,20),method='bounded'); print('free sat tau',q.x, 'fit rate/last strain RMSE',np.sqrt(np.mean((S-U[:,None]*sat(q.x))**2,axis=0))[-2:]); print('median diag residual vs predicted base rate',np.median(S[:,-1]),'U/rate corr',np.corrcoef(U,S[:,-1])[0,1]);
