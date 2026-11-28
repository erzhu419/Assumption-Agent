import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; f,b=m._cosine_operator(s['panel_b']);z=m._decode(tr['x'],m._calibration(s['panel_a']));w=m._affine(np.cos(f*z))@b;a=m._scalar_design(w);p=m._fit(a[:52],y[:52]);print(f,p)
for name,g in [('tanh',np.tanh),('atan',np.arctan),('rational',lambda h:h/(1+h*h)),('sqrt',lambda h:h/np.sqrt(1+h*h))]:
 for mode in [0,1,2]:
  def pred(q):
   h=a@q[:6]; k=np.exp(np.clip(q[6],-4,3)); return h+q[7]*g(k*h) if mode==0 else q[7]*g(k*h)+h+q[8] if mode==1 else h+.22*g(k*h)
  r=min((least_squares(lambda q:(pred(q)-y)[:52],np.r_[p*f,np.log(k),.22,0],max_nfev=500) for k in [.7,1.25,2.15] for f in [.6,1.1]),key=lambda r:r.cost);e=abs(pred(r.x)-y);print(name,mode,e[:52].mean(),e[52:].mean(),r.x)
fit=m._fit(m._affine(w[:52]),T[:52]);print('test cosine range',np.cos(f*m._decode(s['test']['x'],m._calibration(s['panel_a']))).min(axis=0),np.cos(f*m._decode(s['test']['x'],m._calibration(s['panel_a']))).max(axis=0)); print('temporal fit residual');
for n,func in [('log',lambda t,p:np.log1p(p[0]*t)),('exp',lambda t,p:-np.expm1(-p[0]*t)),('sqrt',lambda t,p:np.sqrt(t+p[0])-np.sqrt(p[0]))]:
 for use in [False,True]:
  def B(p):
   col=func(t,p);return np.c_[col,np.sin(p[1]*t)] if use else col[:,None]
  def r(p):
   a=B(p);return (a@m._fit(a,fit[1:].T)-fit[1:].T).ravel()
  res=least_squares(r,[1.25,2.15],bounds=(.001,8));print(n,use,res.x,np.sqrt(np.mean(res.fun**2)))
