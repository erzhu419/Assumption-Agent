import pickle
import numpy as np
from scipy.optimize import least_squares
import prony_rheology as p
x=pickle.load(open('public_rheology.pkl','rb'))
e=x['experiment']; t,s,y,u,v=(e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')); truth=x['gt_test_stress']; d=np.c_[np.ones_like(t),(t-t.mean())/np.ptp(t)]
tau=np.geomspace(.02,20,60); a=p._design(t,s,tau)
for mode in ('huber','cauchy'):
 w=np.ones_like(y)
 for k in range(12):
  g,r=p._projected_fit(a,d,y,w)
  z=r/p._noise(r); w=1/np.hypot(1,z) if mode=='cauchy' else np.sqrt(np.minimum(1,1.5/np.maximum(abs(z),1e-12)))
 mask=abs(r-np.median(r))<=4.5*p._noise(r)
 print(mode,'inliers',mask.sum(),'G',g[0],'stress error',np.linalg.norm(p._design(u,v,tau)@g-truth)/np.linalg.norm(truth))
print('first switch',np.c_[t[275:285],s[275:285],y[275:285]])
def analyze(t,s,y,d,mask,starts):
 dy=d[mask]; yy=y[mask]; q=np.linalg.qr(dy,mode='reduced')[0]; yp=yy-q@(q.T@yy)
 def prof(z, full=False):
  a=p._design(t,s,np.exp(z))[mask]; a-=q@(q.T@a); g=np.linalg.lstsq(a,yp,rcond=1e-12)[0]; r=yp-a@g
  return (g,r) if full else r
 for start in starts:
  fit=least_squares(prof,np.log(start),bounds=(np.log(.02),np.log(20)),max_nfev=100)
  g,r=prof(fit.x,True); tt=np.exp(fit.x)
  rel=np.linalg.norm(p._design(u,v,tt)@g-truth)/np.linalg.norm(truth)
  print('modes',len(tt),'tau',tt.round(5),'G',g.round(5),'rms',np.sqrt(np.mean(r*r)),'rel',rel)
print('train-only unpenalized profile')
analyze(t,s,y,d,mask,[[.1,1,10],[.04,.27,1.55,10],[.04,.27,1.55,6,14],[.04,.27,.6,1.55,10]])
print('public reference-only forward-model diagnostic (never used to estimate parameters for artifact)')
analyze(u,v,truth,np.empty((len(u),0)),np.ones(len(u),bool),[[.04,.27,1.55,10],[.04,.27,1.55,6,14]])
