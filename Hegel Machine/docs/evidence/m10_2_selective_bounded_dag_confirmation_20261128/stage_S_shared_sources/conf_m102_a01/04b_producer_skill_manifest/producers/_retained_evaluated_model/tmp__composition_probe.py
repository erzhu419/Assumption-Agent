import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; t=s['eval_times']; Y=np.c_[s['train']['y'],s['train']['traj']]; c=s['panel_c']; u=c['u']; C=[]
for nm,fun,ini in [('sqrt',lambda u,p:u+p[0]*(np.sqrt(1+p[1]*u*u)-1),[1.2,.92]),('tanh',lambda u,p:np.tanh(p[0]*u),[1.5]),('tanhwarp',lambda u,p:np.tanh(p[0]*u+p[1]*u**3)+p[2]*u,[1.5,0,0]),('tanhsin',lambda u,p:np.tanh(p[0]*u)+p[1]*np.sin(p[2]*u),[1.5,.1,2]),('smooth',lambda u,p:u/np.sqrt(1+p[0]*u*u),[1.5]),('asinh',lambda u,p:np.arcsinh(p[0]*u),[2]),('atan',lambda u,p:np.arctan(p[0]*u),[2]),('tanhshift',lambda u,p:np.tanh(p[0]*(u+np.array(p[1:]))),[1.5,0,0,0,0])]:
 v=c['v2' if nm=='sqrt' else 'v1']; aug=lambda u:np.c_[np.ones(len(u)),u];
 def ff(p):
  F=aug(fun(u,p)); return F,np.linalg.lstsq(F,v,rcond=None)[0]
 def res(p):
  F,W=ff(p); return (F@W-v).ravel()
 r=least_squares(res,ini,max_nfev=200); F,W=ff(r.x); print(nm,r.x,np.std(r.fun),np.linalg.svd(W[1:],compute_uv=False)); C.append((nm,lambda z,fun=fun,p=r.x,W=W:aug(fun(z,p))@W))
np.set_printoptions(precision=4,suppress=True)
print('panel ranges',u.min(0),u.max(0))
for name in ('a','b'):
 panel=s['panel_'+name]; z=panel['u']; v=panel['v']; keep=np.ones(len(z),bool)
 for k in range(4):
  W=np.linalg.lstsq(aug(z[keep]),v[keep],rcond=None)[0]; keep=np.linalg.norm(aug(z)@W-v,axis=1)<2
 if name=='a': W[1:]=np.diag(np.diag(W[1:])); A=W
 else: B=W
trans=[('x',lambda z:z),('a',lambda z:aug(z)@A),('ai',lambda z:(z-A[0])@np.linalg.inv(A[1:])),('b',lambda z:aug(z)@B),('bi',lambda z:(z-B[0])@np.linalg.inv(B[1:]))]; trans += [(p+q,lambda z,f=f,g=g:g(f(z))) for p,f in trans[1:3] for q,g in trans[3:]]
rng=np.random.default_rng(497); order=[np.arange(72)]+[rng.permutation(72) for _ in range(7)]; I=np.array([r[:52] for r in order]); J=np.array([r[52:] for r in order]);
def cv(F,l):
 D=F[I]; P=F[J]; w=np.linalg.solve(D.transpose(0,2,1)@D+l*np.eye(D.shape[-1]),D.transpose(0,2,1)@Y[I]); e=abs(P@w-Y[J]); return np.r_[e[0,:,0].mean(),e[0,:,1:].mean(),e[:,:,0].mean(),e[:,:,1:].mean()]
r=[]
for nm,g in trans:
 z=g(x)
 for spec,f in [('raw',lambda z:z)]+C:
  h=f(z)
  for d in (1,2,3):
   for typ,F in [('poly',PolynomialFeatures(d).fit_transform(h))]+([('mix',np.c_[PolynomialFeatures(2).fit_transform(h),z]),('bothlinear',np.c_[aug(h),x]),('mixquad',np.c_[PolynomialFeatures(2).fit_transform(h),PolynomialFeatures(2).fit_transform(z)[:,1:]])] if d==2 and spec!='raw' else []):
    for l in ([.002,.025,.12] if typ=='mixquad' or d==3 else [.0002,.02]):
     e=cv(F,l); r.append((nm,spec,d,typ,l,e))
for nm,g in trans:
 z=g(x)
 for nm1,g1 in C[:1]:
  for nm2,g2 in C[1:]:
   h=g1(z); k=g2(z); F=np.c_[aug(h),k];
   for d in [1,2]:
    G=PolynomialFeatures(d).fit_transform(F[:,1:]);
    for l in [.002,.025,.12]:r.append((nm,nm1+'+'+nm2,d,'joint',l,cv(G,l)))
for j in [2,3,0,1]:
 print('rank',j)
 for row in sorted(r,key=lambda row:row[-1][j])[:9]:print(row)
