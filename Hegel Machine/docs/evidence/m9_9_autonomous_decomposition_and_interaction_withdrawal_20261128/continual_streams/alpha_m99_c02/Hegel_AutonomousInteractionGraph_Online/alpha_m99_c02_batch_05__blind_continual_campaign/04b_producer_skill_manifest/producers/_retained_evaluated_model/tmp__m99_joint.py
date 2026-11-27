import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import *
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
ds = calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
tr,te,d = (ds[k] for k in ('train','test','chamber_verification_trials'))
dt,t=ds['past_dt'],ds['eval_times']
x,xt,xd=features(tr,te,d)
taus,theta=material(tr,x,dt,t)
print('dt,tau,theta',dt,taus,theta)
from scipy.optimize import least_squares
Y=np.c_[d['instant_rate_obs'],d['strain_curves_obs']-d['strain_0'][:,None]]
P,H=loads(d); D=theta[2]+xd@theta[3:]; Z=H[:,None]-gap_for(taus,P,H,dt); O=np.ones(len(H)); V=np.r_[1,t]; L=d['lab_id']
def modes(q): return np.vstack((1/q,-np.expm1(-t[:,None]/q))).T
def run(name,inputs,free=False):
 def fit(ix):
  def inner(logq,ret=False):
   q=np.exp(logq); A=np.vstack((modes(q),V)); B=np.concatenate([a[None,:,None]*J[:,None,:] for a,J in zip(A,inputs)],axis=2); C=B[ix].reshape(-1,B.shape[-1]); b=np.linalg.lstsq(C,Y[ix].ravel(),rcond=1e-9)[0]; E=B@b-Y
   return (E,b,q) if ret else E[ix].ravel()
  q=np.log(taus); q=least_squares(inner,q,bounds=(np.log([.4,1.5]),np.log([1.1,15.])),max_nfev=150).x if free else q
  return inner(q,True)
 e,b,q=fit(O.astype(bool)); a=np.array([fit(L!=j)[0][L==j] for j in np.unique(L)],dtype=object) if False else [fit(L!=j)[0][L==j] for j in np.unique(L)]
 rms=lambda z:np.round([np.sqrt(np.mean(z[:,0]**2)),np.sqrt(np.mean(z[:,1:]**2))],5)
 print(name,'q',q.round(5),'fit',rms(e),'worst',np.max([rms(z) for z in a],axis=0),'coefs',b.round(5))
for free in (False,True):
 for name,inputs in [('H-D',[np.c_[H,D]]*2+[D[:,None]]),('H-D-x',[np.c_[H,O,xd]]*2+[np.c_[O,xd]]),('H-D+history',[np.c_[H,D,Z]]*2+[D[:,None]]),('slow+history',[np.c_[H,D],np.c_[H,D,Z],D[:,None]]),('slow+steps',[np.c_[H,D],np.c_[H,D,P],D[:,None]]),('curved',[np.c_[H,D,Z,H*D,D**2,H**2]]*2+[np.c_[D,D**2,H*D]]),('slow+steps-x',[np.c_[H,O,xd],np.c_[H,O,xd,P],np.c_[O,xd]])]: run(name,inputs,free)
