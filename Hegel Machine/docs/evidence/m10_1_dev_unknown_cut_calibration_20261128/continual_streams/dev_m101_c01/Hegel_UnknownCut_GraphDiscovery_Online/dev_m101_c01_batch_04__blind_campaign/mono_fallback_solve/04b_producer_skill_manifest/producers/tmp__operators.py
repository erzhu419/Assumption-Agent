from structure import *
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement

def aff(u): return np.c_[np.ones(len(u)),u]
def fit(F,v): return np.linalg.lstsq(F,v,rcond=None)[0]
pc=s['panel_c']; pb=s['panel_b']; bb=fit(aff(pb['u']),pb['v']); u=pc['u']; v=pc['v1']
def c1design(u,p):return aff(np.tanh(u*p[:4]+p[4:]))
def rc(p):
 F=c1design(u,p); return (F@fit(F,v)-v).ravel()
p=least_squares(rc,np.r_[np.ones(4)*1.6,np.zeros(4)],bounds=(np.r_[np.ones(4)*.2,np.ones(4)*-2],np.r_[np.ones(4)*4,np.ones(4)*2])).x
c1=fit(c1design(u,p),v); c1map=lambda a:c1design(a,p)@c1
c2design=lambda a:np.c_[aff(a),a*a]; c2=fit(c2design(u),pc['v2']); c2map=lambda a:c2design(a)@c2
if __name__=='__main__':
 print('C1 scales/shifts',p,'rms',np.std(rc(p))); print('C2 rms',np.std(c2design(u)@c2-pc['v2'])); print('B singular values',np.linalg.svd(bb[1:],compute_uv=False))
 print('mean trajectory',z.mean(0)); g=np.linalg.svd(z-z.mean(0),full_matrices=False)[2][0];print('mean orthogonal',z.mean(0)-g*(g@z.mean(0)));print('primary time',g/g[-1]);
 from scipy.optimize import least_squares
 for name,fn,init in [('exp',lambda q:-np.expm1(-q[0]*t),[.8]),('shifted-exp',lambda q:1-q[1]*np.exp(-q[0]*t),[.8,1]),('exp-sine',lambda q:-np.expm1(-q[0]*t)+q[1]*np.sin(q[2]*t),[.8,.02,1.6]),('biexp',lambda q:-np.expm1(-q[0]*t)+q[1]*-np.expm1(-q[2]*t),[.7,.02,2]),('tanh',lambda q:np.tanh(q[0]*t),[.8])]:
  f=least_squares(lambda q:fn(q)/fn(q)[-1]-g/g[-1],init,max_nfev=300);print(name,f.x,np.std(f.fun))
 base=P[:,dg<=2]; K=base@base.T
 B=aff(x)@bb
 groups={'A':x,'B':B,'Blinear':x@bb[1:],'A.C1':c1map(x),'A.C2':c2map(x),'B.C1':c1map(B),'B.C2':c2map(B)}
 out=[]
 for key,Z in groups.items():
  if key.endswith(('C1','C2')):
   for name,F in [('linear',Z),('quad',PolynomialFeatures(2).fit_transform(Z)),('xprod',(x[:,:,None]*Z[:,None,:]).reshape(len(x),-1)),('Bprod',(B[:,:,None]*Z[:,None,:]).reshape(len(x),-1)),('Bdiag',B*Z),('xdiag',x*Z)]:
    out.append((key,name,metrics(K+F@F.T,1e-5)))
  for f in [np.tanh,np.sin,lambda x:x/(1+x*x)]:
   for scale in [.5,1,1.6,2.5]:
    W=f(Z*scale); matrices={'lin':W,'quad':PolynomialFeatures(2).fit_transform(W),'Zprof':np.c_[W,Z*W],'xprod':np.c_[W,(x[:,:,None]*W[:,None,:]).reshape(len(x),-1)],'linquad':np.c_[W,W*W]}
    for nm,F in matrices.items():out.append((key,f.__name__+str(scale)+nm,metrics(K+F@F.T,1e-5)))
 for j in range(2):
  print('BEST output',j)
  for v in sorted(out,key=lambda v:(v[2][2+j]+v[2][4+j])/2)[:18]: print(v)
