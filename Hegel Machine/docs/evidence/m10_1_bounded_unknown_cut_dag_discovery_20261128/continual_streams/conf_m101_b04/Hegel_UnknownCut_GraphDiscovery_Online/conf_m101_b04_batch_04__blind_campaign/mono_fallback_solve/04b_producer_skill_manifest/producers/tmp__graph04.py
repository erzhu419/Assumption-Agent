from cal04 import s, u, v, fit, aff, trim
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
np.set_printoptions(precision=5,suppress=True,linewidth=120)
def mon(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for i in cwr(range(4),d)])
def ac(a,b):return np.c_[np.ones(len(a)),a,b,a*b,a**3]
def decode(x,u=u,v=v):
 w=[]
 for a,b,y in zip(u.T,np.roll(u,-1,axis=1).T,v.T):
  X=ac(a,b); _,c,r,ix=trim(X,y,44); ix=abs(r)<.001; w.append(fit(X[ix],y[ix]))
 w=np.array(w); print('cal',w)
 z=(x-w[:,0])/w[:,1]; ii=np.arange(4)
 for j in range(20):
  b=np.roll(z,-1,axis=1); c=w.T; r=c[0]+c[1]*z+c[2]*b+c[3]*z*b+c[4]*z**3-x
  J=np.zeros((len(z),4,4)); J[:,ii,ii]=c[1]+c[3]*b+3*c[4]*z*z; J[:,ii,(ii+1)%4]=c[2]+c[3]*z
  z-=np.linalg.solve(J,r[...,None])[...,0]
 print('inverse',np.max(abs(r)),z.min(0),z.max(0));return z
x=s['train']['x']; z=decode(np.r_[x,s['test']['x']]); b=s['panel_b']; wb=fit(aff(b['u']),b['v']); a=z; z=aff(a)@wb
p=s['panel_c']; c=p['u']
def panel_features(x,name,r=1):
 if name=='q':return np.c_[aff(x),mon(x,2)]
 if name=='s':return np.c_[aff(x),np.sin(x*r)]
 if name=='t':return np.c_[aff(x),np.tanh(x*r)]
 if name=='cos':return np.c_[aff(x),np.cos(x*r)]
 if name=='q+s':return np.c_[aff(x),mon(x,2),np.sin(x*r)]
 if name=='cube':return np.c_[aff(x),x**3]
 if name=='pair':return np.c_[aff(x),x*np.roll(x,-1,axis=1)]
 if name=='square':return np.c_[aff(x),x*x]
 if name=='rational':return np.c_[aff(x),x/(1+r*x*x)]
coords={'raw':np.r_[x,s['test']['x']], 'a':a,'ab':z}
for k in ['v1','v2']:
 v=p[k]; best=[]
 for name in ['q','s','t','cos','q+s','cube','pair','square','rational']:
  def calc(r):
   F=panel_features(c,name,r[0]); return (F@fit(F,v)-v).ravel()
  rr=min((least_squares(calc,[start],bounds=(.1,8)) for start in [.5,1.7,3.5]),key=lambda o:sum(o.fun**2)).x[0] if name in ['s','t','cos','q+s','rational'] else 1.
  F=panel_features(c,name,rr); W=fit(F,v); R=v-F@W; h=(np.linalg.qr(F)[0]**2).sum(1); E=np.sqrt(np.mean((R/(1-h[:,None]))**2)); best.append((E,name,rr,W)); print('c',k,name,rr,np.sqrt(np.mean(R**2)),E)
 _,name,rr,W=min(best,key=lambda e:e[0]); print(k,'winning coefs',W); coords[k+'(a)']=panel_features(a,name,rr)@W; coords[k+'(ab)']=panel_features(z,name,rr)@W
T=s['eval_times']; Y=s['train']['y']; S=s['train']['traj']; B=lambda r:-np.expm1(-T[:,None]*r); tr=slice(0,52); te=slice(52,72)
def res(r):
 f=B(r); return (f@fit(f,S[tr].T)-S[tr].T).ravel()
r=least_squares(res,[1.8,.4]).x; basis=np.linalg.qr(B(r))[0]; target=np.c_[Y,S@basis]; print('r',r)
for name,zz in coords.items():
 for n,F in [('lin',aff(zz)),('sep',np.c_[aff(zz),zz**2,zz**3]),('full2',np.c_[aff(zz),mon(zz,2)]),('cubic',np.c_[aff(zz),mon(zz,2),mon(zz,3)]),('pair',np.c_[aff(zz),zz*np.roll(zz,-1,axis=1)]),('sine',np.c_[aff(zz),np.sin(1.5*zz)])]:
  W=fit(F[tr],target[tr]); P=F[:72]@W; E=[np.mean(abs(P[idx,0]-Y[idx])) for idx in [tr,te]]+[np.mean(abs(P[idx,1:]@basis.T-S[idx])) for idx in [tr,te]]; print(name,n,np.round(E,5))
  if name in ['a','ab'] and n=='full2':print('full2 Coeffs',W)
