import sys,pickle,importlib.util,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_batch_02_solver.py'); prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; c=calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics'])); tr,te,q=(c[k] for k in ('train','test','chamber_verification_trials')); t,dt=(c[k] for k in ('eval_times','past_dt'))
x,xt=prior._features(tr,te); _,xq=prior._features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)
p,h=prior._loads(tr); pq,hq=prior._loads(q); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,beta=prior._fit(x,p,h,dt,t,y,r)
yq=q['strain_curves_obs']-q['strain_0'][:,None]; rq=q['instant_rate_obs']; drift=beta[2]+xq@beta[3:]; Y=np.column_stack((yq,rq));
def summary(e): return np.round(np.percentile(np.abs(e),[50,80,95,100],axis=None),6)
for name,vals in [('train',prior._design(tau,x,p,h,dt,t)),('diag_bench',prior._design(tau,xq,pq,hq,dt,t))]:
 a,b=vals; yy,rr=(y,r) if name=='train' else (yq,rq); print(name,summary(a@beta-yy),summary(b@beta-rr))
for nn in [1,2]:
 path=f'/tmp/hegel_continual_archive/alpha_m99_c01_batch_0{nn}_solver.py'; sp=importlib.util.spec_from_file_location('prev',path); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 a,b=m._chamber(tau,beta,xq,hq,t) if nn==1 else m._chamber(tau,beta,xq,pq,hq,dt,t); print('prior',nn,summary(a-yq),summary(b-rq))
print('material:',tau,beta)

def modes(ts):
 return np.column_stack((np.vstack((-np.expm1(-t[:,None]/ts),1./ts)),np.r_[t,1.]))
def prof(logts, data=Y, cutoff=.004):
 a=modes(np.sort(np.exp(logts))); coef=np.linalg.lstsq(a,data.T,rcond=None)[0];
 for it in range(10):
  e=(a@coef-data.T); ww=(1+(e/cutoff)**2)**(-.25); coef=np.stack([np.linalg.lstsq(a*w[:,None],v*w,rcond=None)[0] for v,w in zip(data,ww.T)],axis=1)
 return (a@coef-data.T).T,coef.T
if __name__=='__main__':
 for n,ss in [(1,[[.2],[.8],[3]]),(2,[[.1,1],[.3,2],[.6,3],[1,8],tau]),(3,[[.1,.5,2],[.2,1,5]])]:
  fits=[least_squares(lambda l:prof(l)[0].ravel(),np.log(s),bounds=(np.log(.045),np.log(30)),loss='soft_l1',f_scale=.006,max_nfev=100,diff_step=.0001) for s in ss]; f=min(fits,key=lambda f:f.cost); ts=np.sort(np.exp(f.x)); e,b=prof(f.x); print('free',n,ts,'cost',f.cost,'curves',summary(e[:,:-1]),'rate',summary(e[:,-1]));
  if n==2:
   inp=np.column_stack((hq,pq,drift,xq,prior._gap(tau,pq,hq,dt),prior._gap(ts,pq,hq,dt))); co=np.corrcoef(inp.T,b.T)[:-3,-3:]; print('linear correlations: hold, past, drift, x, gap bench, gap chamber'); print(np.round(co,2)); print('coefs sample',np.round(b[:8],3));
 print('diag first rows',np.round(Y[:3],4))
