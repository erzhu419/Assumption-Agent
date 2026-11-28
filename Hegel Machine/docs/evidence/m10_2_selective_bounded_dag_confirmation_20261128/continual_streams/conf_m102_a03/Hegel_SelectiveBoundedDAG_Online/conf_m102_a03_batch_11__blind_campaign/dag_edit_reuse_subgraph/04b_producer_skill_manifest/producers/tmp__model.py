import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr

def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-11)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,p],axis=1) for k in range(1,d+1) for p in cwr(range(x.shape[1]),k)])
def coords(S):
    U,V=(S['panel_a'][k] for k in ('u','v')); good=np.linalg.norm(V,axis=1)<5; u,v=U[good],V[good]; x=np.r_[S['train']['x'],S['test']['x']]
    def des(w,j,h):
        a=w[:,j]; b=w[:,(j+1)%4]; c=w[:,(j+2)%4]
        return np.c_[aff(w),a/(1+h[0]*a*a),np.sin(h[1]*c),a*b]
    def coef(h): return [fit(des(u,j,h),v[:,j]) for j in range(4)]
    def fw(w,h,co):return np.column_stack([des(w,j,h)@co[j] for j in range(4)])
    h=least_squares(lambda h:(fw(u,h,coef(h))-v).ravel(),[1.25,2.15],bounds=([.1,.3],[5,4])).x; co=coef(h)
    g=fit(aff(v),u); z=np.array([least_squares(lambda w:fw(w[None],h,co)[0]-a,b,gtol=1e-12).x for a,b in zip(x,aff(x)@g)])
    B=S['panel_b']; b=aff(z)@fit(aff(B['u']),B['v']); C=S['panel_c']; u=C['u'];
    def p1(w,h):return np.c_[aff(np.arctan(h[0]*w)),w]
    h1=least_squares(lambda h:(p1(u,h)@fit(p1(u,h),C['v1'])-C['v1']).ravel(),[2.15],bounds=([.2],[6])).x
    def p2(w):return np.c_[aff(w),w*w,w**3]
    def c1(w): return p1(w,h1)@fit(p1(u,h1),C['v1'])
    def c2(w): return p2(w)@fit(p2(u),C['v2'])
    return {'z':z,'b':b,'c1z':c1(z),'c2z':c2(z),'c1b':c1(b),'c2b':c2(b),'x':x}, (h,np.std(fw(U[good],h,co)-V[good]),h1,np.std(c1(u)-C['v1']),np.std(c2(u)-C['v2']))
if __name__=='__main__':
 import pickle
 S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; D,diag=coords(S); print('panel fits',diag); y=S['train']['y']; t=S['train']['traj']; n=len(y)
 F=dict(D)
 for name in ('z','b','c1z','c1b'):
  w=D[name]
  for k,v in {'a':np.arctan(2.15*w),'s':np.sin(2.15*w),'r':w/(1+1.25*w*w),'t':np.tanh(1.25*w),'p':w/(1+.22*w*w)}.items(): F[name+k]=v
 rng=np.random.default_rng(2); groups=[np.arange(52,n)]+[np.sort(rng.choice(n,20,False)) for _ in range(3)]
 def val(A):
  r=[]
  for j in groups:
   i=np.setdiff1d(np.arange(n),j); R=A[i]; P=A[j]@np.linalg.pinv(R,rcond=1e-9); r.append([np.mean(abs(P@y[i]-y[j])),np.mean(abs(P@t[i]-t[j]))]);
  r=np.array(r); return np.r_[r[0],r[1:].mean(0),r[1:].max(0)]
 res=[]
 def chk(s,A):res.append((val(A),s))
 for k,w in F.items():
  for d in (1,2,3): chk((k,d),poly(w,d))
 for k,k2 in cwr(F,2):
  if k==k2:continue
  w,v=F[k],F[k2]; chk((k,k2,'lin'),aff(np.c_[w,v])); chk((k,k2,'pair'),aff(np.c_[w,v,(w[:,:,None]*v[:,None,:]).reshape(len(w),-1)])); chk((k,k2,'sumquad'),np.c_[poly(w,2),poly(v,2)[:,1:]])
 for a in (0,1):
  print('OBJECTIVE',a)
  for V,K in sorted(res,key=lambda r: .45*r[0][a]+.55*r[0][a+2])[:18]: print(K,V.round(5))
 np.savez('/tmp/features.npz',**F)
