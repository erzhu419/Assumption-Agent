import numpy as np
from scipy.optimize import nnls

def mem(t,e,taus):
 t=np.asarray(t,float); e=np.asarray(e,float); dt=np.diff(t)
 if np.any(dt<=0): raise ValueError('time must be strictly increasing')
 r=np.diff(e)/dt; D=np.zeros((len(t),len(taus)))
 for i,(h,v) in enumerate(zip(dt,r),1): D[i]=D[i-1]*np.exp(-h/taus)+v*taus*(-np.expm1(-h/taus))
 return D

def fit(t,e,y,taus):
 D=np.c_[e,mem(t,e,taus)]; Q=np.c_[np.ones(len(t)),t]
 P=lambda z: z-Q@np.linalg.lstsq(Q,z,rcond=None)[0]
 B=P(D); z=P(y); w=np.ones(len(y)); b=None
 for _ in range(12):
  sw=np.sqrt(w); norms=np.maximum(np.linalg.norm(B*sw[:,None],axis=0),1e-12)
  c,_=nnls(B*sw[:,None]/norms,z*sw,maxiter=5000); b=c/norms
  r=y-D@b-Q@np.linalg.lstsq(Q,y-D@b,rcond=None)[0]
  s=1.4826*np.median(abs(r-np.median(r)))+1e-10
  nw=np.minimum(1,1.345*s/np.maximum(abs(r),1e-12))
  if np.max(abs(nw-w))<1e-5: break
  w=nw
 return b
if __name__=='__main__':
 import pickle
 p=pickle.load(open('/holdout_task/public_rheology.pkl','rb'));e=p['experiment']; y=p['gt_test_stress']
 for n in (6,8,10,12,16,20,32,48):
  ts=np.geomspace(.02,20,n);b=fit(e['train_t'],e['train_strain'],e['train_stress_raw'],ts)
  pred=np.c_[e['test_strain'],mem(e['test_t'],e['test_strain'],ts)]@b
  print(n,np.sqrt(np.mean((pred-y)**2)/np.mean(y*y)),b[0],abs(b[0]-p['gt_G_inf'])/p['gt_G_inf'])
