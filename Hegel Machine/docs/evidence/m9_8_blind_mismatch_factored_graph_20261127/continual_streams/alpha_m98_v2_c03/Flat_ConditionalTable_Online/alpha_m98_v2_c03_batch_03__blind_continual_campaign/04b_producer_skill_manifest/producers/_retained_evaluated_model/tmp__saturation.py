import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=i['campaign_diagnostics']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; train,diag=m._prepare(dict(d,test=c)); y=np.column_stack([tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]); tau,theta=m._fit(train,dt,t,y)
yc=np.column_stack([c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]); pred=m._design(diag,dt,t,tau)@theta; r=yc-pred; past,h,f=diag; drift=f@theta[2:]; g=m._design(diag,dt,t,tau)[:,-1,:2]*tau; z=h[:,None]-g
rm=lambda a:np.sqrt(np.mean(a*a,axis=0))
from scipy.optimize import least_squares
lab=c['lab_id']; F=np.column_stack([np.ones(len(h)),f[:,1:]]); wt=np.r_[np.ones(len(t)),.8]; E=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); T=np.r_[t,1.]; base=pred-drift[:,None]*T
err=lambda e: np.sqrt(np.mean(e**2))
for label in ['exp','rational','rational2','tanh','exp+linear']:
 def sig(p):
  b=np.exp(p[0]); u=b*t
  if label.startswith('exp'): s=-np.expm1(-u)/b
  if label=='rational': s=t/(1+u)
  if label=='rational2': s=t/(1+u*u)
  if label=='tanh': s=np.tanh(u)/b
  return np.r_[s,1.]
 for correction in ['none','intercept','features','memory+features']:
  def calc(p):
   if correction=='none': amp=p[1]*drift
   elif correction=='intercept': amp=p[1]*drift+p[2]
   else: amp=F@p[1:6]
   out=base+amp[:,None]*sig(p)
   if label=='exp+linear': out+=p[-1]*drift[:,None]*T
   if correction=='memory+features': out+=((np.c_[np.ones(len(h)),z] @ p[6:12].reshape(3,2)) @ E.T)
   return out
  n={'none':2,'intercept':3,'features':6,'memory+features':12}[correction]; seed=np.r_[np.log(.8),1.5,np.zeros(n-2+int(label=='exp+linear'))];
  def fit(q): return least_squares(lambda p: ((calc(p)-yc)[q]*wt).ravel(),seed,max_nfev=100, bounds=(np.r_[-5,np.full(len(seed)-1,-np.inf)],np.r_[3,np.full(len(seed)-1,np.inf)])).x
  p=fit(np.ones(len(h),bool)); e=calc(p)-yc; cv=[]
  for L in np.unique(lab):
   q=lab==L; o=(calc(fit(~q))-yc)[q]; cv.append((err(o[:,:-1]),err(o[:,-1])))
  print(label,correction,'params',p[:6].round(4),'train',np.round([err(e[:,:-1]),err(e[:,-1])],4),'worst LOLO',np.round(np.max(cv,axis=0),4))
