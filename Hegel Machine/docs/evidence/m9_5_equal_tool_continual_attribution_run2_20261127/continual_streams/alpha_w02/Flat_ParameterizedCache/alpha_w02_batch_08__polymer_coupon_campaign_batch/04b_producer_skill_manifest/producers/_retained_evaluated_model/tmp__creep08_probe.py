import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/alpha_w02_batch_07_solver.py';s=importlib.util.spec_from_file_location('prior',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w02_batch_08.pkl','rb'))[0]['dataset'];a,b=d['train'],d['test'];t,dt=d['eval_times'],d['past_dt'];sy=d['yield_stress'];f=m._features(a,b)[:len(a['lab_id'])];y=np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']];complete=np.isfinite(a['coupon_features_raw']).all(axis=1);lab=a['lab_id'];rng=np.random.default_rng(203);val=np.zeros(len(lab),bool)
for j in range(4):
 for k in np.unique(a['stratum']):
  ii=np.flatnonzero((lab==j)&(a['stratum']==k)&complete);val[rng.choice(ii,max(1,len(ii)//4),replace=False)]=True
fit=complete&~val; anchor=np.polyfit(d['calibration_load_cells']['cmd_stress'],d['calibration_load_cells']['true_stress'],1); old=np.vstack((anchor, [[1.25223,.14514],[1.23634,.1322],[.76306,-.14437]])); h=np.r_[t,1.]; q=np.linalg.qr(t[:,None],mode='complete')[0][:,1:];
def design(z):
 cal=np.vstack((anchor,z[2:].reshape(3,2)));tau=np.exp(z[:2]);return m._design(m._inputs(a,f,cal,dt,sy),dt,t,tau)
def fit_model(initial, mode):
 def layout(w):
  if mode=='rate':return w[:,-1]
  if mode=='curvature+rate':return np.concatenate((np.einsum('tk,nt...->nk...',q,w[:,:-1]),3*w[:,-1:]),axis=1)
  return w
 target=layout(y[fit]).ravel();w=np.ones_like(target)
 def fun(z,return_model=False):
  X=layout(design(z)[fit]).reshape(-1,9); c=np.linalg.lstsq(X*w[:,None],target*w,rcond=None)[0]; e=X@c-target
  return (c,e) if return_model else w*e
 z=np.r_[np.log([.55,3.9]), initial[1:].ravel()];lo=np.r_[np.log([.051,.7]),[.4,-.6]*3];hi=np.r_[np.log([3.,24.99]),[2, .6]*3]
 for _ in range(5):
  opt=least_squares(fun,z,bounds=(lo,hi),max_nfev=90);z=opt.x;c,e=fun(z,True); rr=e.reshape(fit.sum(),-1); dev=np.maximum(np.median(abs(rr-np.median(rr,axis=0)),axis=0)*1.4826,2e-4);w=(1+(rr/(3*dev))**2).ravel()**-.5
 return z,c

def report(tag,P):
 e=P-y
 print(tag,'RMSE by time',np.round(np.sqrt(np.mean(e[val]**2,axis=0)),5));print('MAE by time',np.round(np.median(abs(e[val]),axis=0),5))
 print('lab metrics',[(j,*np.round([np.sqrt(np.mean(e[val&(lab==j),:-1]**2)),np.sqrt(np.mean(e[val&(lab==j),-1]**2))],5)) for j in range(4)])
 if tag=='old profile':
  print('curve residual row examples',np.round(e[val][:12],3));print('curve residual quantiles by column',np.round(np.quantile(e[val],[.1,.5,.9],axis=0),4))
I=m._inputs(a,f,old,dt,sy);tau,c=m._fit(tuple(v[fit] for v in I),dt,t,y[fit]);print('old',tau,c); report('old profile',m._design(I,dt,t,tau)@c)
for mode in ['rate','curvature+rate','joint']:
 z,c=fit_model(old,mode);print(mode,np.exp(z[:2]),z[2:].reshape(-1,2),c);report(mode,design(z)@c)
