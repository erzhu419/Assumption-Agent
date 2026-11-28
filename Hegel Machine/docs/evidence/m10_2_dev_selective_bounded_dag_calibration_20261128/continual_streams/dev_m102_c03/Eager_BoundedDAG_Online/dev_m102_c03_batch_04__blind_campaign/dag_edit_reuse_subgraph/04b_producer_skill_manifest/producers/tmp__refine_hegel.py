import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['panel_c']; u=c['u']; w=c['v2']; print('telemetry',s['telemetry_vec'])
funs={'cos':lambda z,k: np.cos(z*k), 'gauss':lambda z,k:np.exp(-k*z*z),'lorentz':lambda z,k:1/(1+k*z*z),'tanh':lambda z,k:np.tanh(z*k),'atan':lambda z,k:np.arctan(z*k),'absExp':lambda z,k:np.exp(-k*abs(z)),'tanh2':lambda z,k:np.tanh(k*z*z),'cos2':lambda z,k:np.cos(k*z*z),'log':lambda z,k:np.log1p(k*z*z),'gauss4':lambda z,k:np.exp(-k*z**4),'sinc':lambda z,k:np.sinc(k*z)}
for name,fn in funs.items():
 out=[]
 for linear in (False,True):
  def mat(p,u):
   h=fn(u-p[4:],np.exp(p[:4])); return np.c_[np.ones(len(u)),h,u] if linear else np.c_[np.ones(len(u)),h]
  def err(p):
   h=mat(p,u); return (h@np.linalg.lstsq(h,w,rcond=1e-9)[0]-w).ravel()
  best=None
  for a in (.3,1,2,3,5):
   for offset in (0,.12,-.12):
    o=least_squares(err,np.r_[np.full(4,np.log(a)),np.full(4,offset)],bounds=([-5]*4+[-.8]*4,[3]*4+[.8]*4),max_nfev=120)
    e=np.sqrt(np.mean(o.fun**2));
    if best is None or e<best[0]: best=e,np.round(np.exp(o.x[:4]),6),np.round(o.x[4:],6)
  out.append(best)
 print(name,out,flush=True)
