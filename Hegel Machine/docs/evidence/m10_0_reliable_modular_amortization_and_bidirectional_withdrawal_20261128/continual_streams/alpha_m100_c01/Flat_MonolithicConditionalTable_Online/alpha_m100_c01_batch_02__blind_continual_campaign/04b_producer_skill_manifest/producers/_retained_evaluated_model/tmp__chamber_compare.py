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
from scipy.optimize import minimize_scalar
from itertools import product

def exp_response(tau): return -tau*np.expm1(-t/tau)
def rat_response(tau): return t/(1+t/tau)
def root_response(tau): return t/np.sqrt(1+(t/tau)**2)
def tanh_response(tau): return tau*np.tanh(t/tau)

for name,f in [('exp',exp_response),('rational',rat_response),('root rational',root_response),('tanh',tanh_response)]:
 for nuisance in [False,True]:
  for cv in ['lab','stratum','missing']:
   group={'lab':g['lab_id'],'stratum':g['stratum'],'missing':np.isnan(g['coupon_features_raw']).any(axis=1)}[cv]
   def fit(mask):
    def calc(z):
     pred=U[:,None]*(f(z)-t)
     return pred, ((pred[mask]-S[mask,:-1])**2).mean()
    if nuisance:
     def calc(z):
      response=f(z); score=U[mask,None]*response; r=S[mask,:-1]+U[mask,None]*t
      k=(np.sum(r*score)+np.sum((S[mask,-1]+U[mask])*U[mask]))/(np.sum(score**2)+np.sum(U[mask]**2)); pred=k*U[:,None]*response-U[:,None]*t
      return pred, ((pred[mask]-S[mask,:-1])**2).mean()+((k-1)*U[mask]-S[mask,-1]).dot((k-1)*U[mask]-S[mask,-1])/np.count_nonzero(mask)/len(t),k
    q=minimize_scalar(lambda z:calc(z)[1],bounds=(.1,20),method='bounded'); out=calc(q.x); return q.x,out[0],out[2] if nuisance else 1.
   err=np.empty_like(S); params=[]
   for v in np.unique(group):
    i=group==v; q,pr,k=fit(~i); params.append((q,k)); err[i,:-1]=pr[i]-S[i,:-1]; err[i,-1]=(k-1)*U[i]-S[i,-1]
   worst=np.max([np.sqrt(np.mean(err[group==v]**2,axis=0)) for v in np.unique(group)],axis=0)
   if not nuisance or name=='exp': print(name,nuisance,cv,'fit',np.array(params).round(5).tolist(),'worst curve/rate',worst[:-1].max(),worst[-1]);
  if name=='exp':
   q,pr,k=fit(np.ones(len(U),bool)); e=np.c_[pr-S[:,:-1],(k-1)*U-S[:,-1]]; print('full exp',q,k,'rmse',np.sqrt(np.mean(e**2,axis=0)).round(5),'residual correlations',np.corrcoef(np.c_[e,H,P,U,F[:,1:]].T)[:e.shape[1],e.shape[1]:][[-2,-1]].round(3))
