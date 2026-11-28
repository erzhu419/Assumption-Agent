import pickle, importlib.util, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; z=tr['traj']; y=tr['y']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def basis(p): return -np.expm1(-t[:,None]*p)
p=least_squares(lambda p:(basis(p)@ls(basis(p),z.T)-z.T).ravel(),[2.15,.4],bounds=([.5,.005],[5.,1.])).x
amp=ls(basis(p),z.T).T; print('rates',p,'rmse',np.std(amp@basis(p).T-z),'amp stats',amp.mean(0),amp.std(0),'corr',np.corrcoef(np.c_[amp,y].T));
f='/tmp/hegel_continual_archive/conf_m102_a02_batch_09_solver.py';sp=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m); cal=m._calibrate(s['panel_a']); u=s['panel_a']['u']; p,b,d=cal; er=b+d*u+p[2]*np.tanh(p[0]*u)+p[3]*np.sin(p[1]*np.roll(u,2,axis=1))-s['panel_a']['v']; print('cal residual',np.sort(np.linalg.norm(er,axis=1))); x=m._inverse(tr['x'], cal);print('latent ranges',x.min(0),x.max(0));
for k in ('panel_b','panel_c'):
 u=s[k]['u']; print(k)
 for name,v in s[k].items():
  if name=='u':continue
  print(name,'v range',v.min(0),v.max(0)); print('aff',ls(np.c_[np.ones(len(u)),u],v));
  for j in range(4):
   def A(q):return np.c_[np.ones(len(u)),u]
   for wh,fn in [('tanh',np.tanh),('sin',np.sin),('soft',lambda q:q/(1+q*q)),('exp',lambda q:q*np.exp(-q*q))]:
    def rr(q):return q[5]+q[6]*fn(A(q)@q[:5])-v[:,j]
    w=ls(A(None),v[:,j]); fits=[least_squares(rr,np.r_[w[0]*a,w[1:]*a,0.,1/a],max_nfev=500) for a in (.7,1.4,3.)]; fit=min(fits,key=lambda f:f.cost); print(j,wh,round(np.std(fit.fun),5),fit.x)
np.savez('/tmp/detail.npz',x=x,amp=amp,rates=p if False else least_squares(lambda p:(basis(p)@ls(basis(p),z.T)-z.T).ravel(),[2.15,.4],bounds=([.5,.005],[5.,1.])).x)
