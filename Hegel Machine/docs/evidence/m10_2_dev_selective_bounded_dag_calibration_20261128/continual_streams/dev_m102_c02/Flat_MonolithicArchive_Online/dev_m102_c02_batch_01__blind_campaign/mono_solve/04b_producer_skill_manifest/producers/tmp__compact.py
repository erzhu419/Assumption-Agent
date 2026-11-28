import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];d=s['panel_b'];u,v=d['u'],d['v'];a=s['panel_a'];p,q=a['u'],a['v'];ok=np.max(abs(q-p-np.median(q-p,0)),1)<1;b,g=np.array([np.linalg.lstsq(np.c_[np.ones(sum(ok)),p[ok,i]],q[ok,i],rcond=None)[0] for i in range(4)]).T
tr=s['train'];z=(tr['x']-b)/g;y=np.c_[tr['y'],tr['traj']]; np.set_printoptions(precision=4,suppress=True,linewidth=120)
def aff(x):return np.c_[np.ones(len(x)),x]
def scoring(A,V):
 H=np.linalg.pinv(A);r=V-A@(H@V);h=(H.T*A).sum(1);return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/np.maximum(1-h[:,None],1e-9))**2))
def feats(w):
 i,j=np.triu_indices(w.shape[1]);return w[:,i]*w[:,j]
def cv(w):
 for a in [aff(w), aff(np.c_[w,feats(w)]),aff(np.c_[w,z,feats(w)])]:
  e=abs(a[52:]@np.linalg.lstsq(a[:52],y[:52],rcond=None)[0]-y[52:]); print(np.round([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()],5),end=' ')
 print()
for deg in range(2,6):
 M=np.hstack([np.ones((len(u),1))]+[u**i for i in range(1,deg+1)]);B=np.linalg.lstsq(M,v,rcond=None)[0];print('ADD',deg,scoring(M,v),'SVDs',np.linalg.svd(v-M@B,compute_uv=False));print(B)
print('forward/backward calibration');
for x in [tr['x'],z,tr['x']*g+b]:
 for fun in [lambda a:a**3,lambda a:np.c_[a,a**3],lambda a:np.c_[a,a**2,a**3,a**4,a**5]]:
  A=aff(fun(x));E=abs(A[52:]@np.linalg.lstsq(A[:52],y[:52],rcond=None)[0]-y[52:]);print(E[:,0].mean(),E[:,1:].mean())
funcs={
'pow':(lambda x,p: np.sign(x)*abs(x)**p[0], [3], [.2],[8]),
'powlinear':(lambda x,p: np.sign(x)*abs(x)**p[0]+p[1]*x,[3,.02],[.2,-3],[8,3]),
'powshift':(lambda x,p: np.sign(x+p[1])*abs(x+p[1])**p[0],[3,0],[.2,-.8],[8,.8]),
'powerscale':(lambda x,p: abs(x)**p[0]*(x+p[1])+p[2]*x,[2,0,.0],[.1,-2,-3],[8,2,3]),
'poly35':(lambda x,p: x**3+p[0]*x+p[1]*x**2+p[2]*x**4+p[3]*x**5,[0]*4,[-5]*4,[5]*4),
'cubicrat':(lambda x,p: (x**3+p[0]*x+p[1]*x*x)/(1+p[2]*x*x),[0,0,.1],[-4,-4,-.3],[4,4,4]),
'cubictanh':(lambda x,p: x**3+p[0]*np.tanh(p[1]*x),[.1,1],[-4,.1],[4,10]),
'cubicexp':(lambda x,p: x**3*np.exp(p[0]*x)+p[1]*x,[0,.1],[-3,-4],[3,4]),
'cubicsoft':(lambda x,p: x**3+p[0]*np.logaddexp(0,p[1]*x)+p[2]*x,[.1,1,.0],[-4,.1,-4],[4,10,4])}
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[pn]['u'],s[pn][vn]; print(pn,vn)
 tests=funcs if pn=='panel_b' else {'shiftedtanh':(lambda x,p:np.tanh(np.exp(p[0])*(x+p[1])) , [.4,0],[-4,-2],[4,2]), 'exp':(lambda x,p:np.expm1(p[0]*x)+p[1]*x,[.9,0],[.05,-4],[3,4]),'squared':(lambda x,p:x+p[0]*x*x+p[1]*x**3,[.45,0],[-4,-4],[4,4])}
 for name,(f,p,lo,hi) in tests.items():
  n=len(p)
  for shared in [True,False]:
   def mat(x,q):return aff(f(x,q if shared else q.reshape(n,4)))
   def res(q):
    A=mat(u,q);return (v-A@np.linalg.lstsq(A,v,rcond=None)[0]).ravel()
   rep=lambda l: np.asarray(l) if shared else np.repeat(l,4)
   r=least_squares(res,rep(p),bounds=(rep(lo),rep(hi)),max_nfev=400,ftol=1e-10);A=mat(u,r.x); B=np.linalg.lstsq(A,v,rcond=None)[0];w=mat(z,r.x)@B
   print(name,shared,np.round(scoring(A,v),6),r.x,end='  ');cv(w)
