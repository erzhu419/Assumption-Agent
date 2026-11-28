from cv06 import *
from scipy.optimize import least_squares
base=np.c_[np.ones(len(z)),z*np.roll(c,1,axis=1),c*d,c,d]
T=np.c_[y,t]
def features(p):
 a,b,k,l=p
 return np.c_[base,np.cos(k*(z[:,0]+a*z[:,1])),np.cos(l*(z[:,2]+b*z[:,3]))]
def fit(a,b,start=(-.5,1,1.25,1.25)):
 def res(p):
  A=features(p)[a];return (A@m._ls(A,T[a])-T[a]).ravel()
 q=least_squares(res,start,bounds=([-3,-3,.1,.1],[3,3,4,4]),gtol=1e-11).x
 F=features(q); C=m._ls(F[a],T[a]); return q,C,F@C
for p in splits:
 a,b=p[:52],p[52:]; q,C,P=fit(a,b); E=abs(P-T); print(q,np.r_[E[a,0].mean(),E[a,1:].mean(),E[b,0].mean(),E[b,1:].mean(),E[b,1:].mean(0).max()]);
 if b[0]==52:
  print('coefs',C.round(4)); print('res svd',np.linalg.svd((P-T)[a],compute_uv=False)); np.savez('/tmp/comp06.npz',base=features(q),res=T-P)
