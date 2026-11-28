import sys; sys.path.insert(0,'/holdout_task')
import batch_solver as m
import itertools,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52; ts=s['eval_times']; np.set_printoptions(precision=5,suppress=True)
z=m._calibrate(s['panel_a'],tr['x']); p=s['panel_b']; u,v=p['u'],p['v']; I=m._intercept; ls=m._ls
c=s['panel_c']; C=lambda x:I(x*np.roll(x,-1,axis=1)); cc=ls(C(c['u']),c['v1'])
def report(w):
 q=C(w)@cc; cross=(z[:,:,None]*q[:,None,:]); a=I(np.c_[z,q,np.trace(cross,axis1=1,axis2=2)]); b=I(np.c_[z,q,cross.reshape(len(z),-1)]); y=tr['y']; t=tr['traj']; cy=ls(a[:n],y[:n]); ct=ls(b[:n],t[:n]); return [np.mean(abs(a[n:]@cy-y[n:])),np.mean(abs(b[n:]@ct-t[n:])),np.std(a[:n]@cy-y[:n]),np.std(b[:n]@ct-t[:n])]
print('baseline', report(m._b_operator(p,z))); print('times',ts); print('mean trajectory',tr['traj'].mean(0)); print('SVD modes',np.linalg.svd(tr['traj']-tr['traj'].mean(0),full_matrices=False)[2][:2]);
for name, fn, g in [('sin',np.sin,2.9),('erf',erf,1.45),('rational',lambda x:x/(1+x*x),.9),('asinh',np.arcsinh,1.8),('tanh',np.tanh,1.45),('gauss',lambda x:x*np.exp(-x*x),1.04)]:
 for mode in ('cyc','mix','mixbias','cubic','sin2','diagbias'):
  def D(a,p):
   scale=np.exp(p[:4]); W=np.diag(scale); j=np.arange(4)
   if mode=='cyc': W[(j+1)%4,j]=p[4:8]
   if mode.startswith('mix'): W[~np.eye(4,dtype=bool)]=p[4:16]
   shift=p[-4:] if mode in ('mixbias','diagbias') else 0
   t=fn(a@W+shift)
   if mode=='cubic':t=np.c_[t,a**3]
   if mode=='sin2':t=np.c_[t,np.sin(a*p[4])]
   return I(np.c_[a,t]);
  def obj(p):d=D(u,p);return (d@ls(d,v)-v).ravel()
  extra={'cyc':4,'mix':12,'mixbias':16,'cubic':0,'sin2':1,'diagbias':4}[mode]
  init=np.r_[np.full(4,np.log(g)),np.zeros(extra)];
  if mode=='sin2':init[-1]=1.25
  f=least_squares(obj,init,max_nfev=150,bounds=(np.r_[np.full(4,-2.),np.full(extra,-5.)],np.r_[np.full(4,1.6),np.full(extra,5.)])); d=D(u,f.x); coef=ls(d,v); print(name,mode,'panel',np.sqrt(np.mean(f.fun**2)),'cv',np.round(report(D(z,f.x)@coef),6),'q',np.round(f.x,3))
