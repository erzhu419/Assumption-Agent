import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; n=len(y)
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(D,y): return np.linalg.lstsq(D,y,rcond=None)[0]
def poly(x,k=2):
 return np.column_stack([np.prod(x[:,ii],axis=1) for j in range(k+1) for ii in itertools.combinations_with_replacement(range(x.shape[1]),j)])
u,v=[s['panel_a'][k] for k in ('u','v')]; z=u.copy()
for j in range(4):
 d=aff(u[:,j:j+1]); w=ls(d,v[:,j]);
 for _ in range(8):
  ids=np.argsort(abs(d@w-v[:,j]))[:36]; w=ls(d[ids],v[ids,j]);
 z[:,j]=u[:,j]*w[1]+w[0]
g=np.std(z,axis=0)/np.std(u,axis=0); a=np.mean(z-u*g,axis=0); x=(tr['x']-a)/g
p=s['panel_b']; W=ls(aff(p['u']),p['v']); b=aff(x)@W
from scipy.special import erf,hyp2f1
from scipy.optimize import least_squares
from collections import defaultdict
F={'atan':lambda a,p:np.arctan(a),'asinh':lambda a,p:np.arcsinh(a),'erf':lambda a,p:erf(a),'tanh':lambda a,p:np.tanh(a),'hyp':lambda a,p:a*hyp2f1(.5,p,1.5,-a*a),'rat':lambda a,p:a/(1+a*a)**p,'sin':lambda a,p:np.sin(a),'soft':lambda a,p:a/(1+abs(a))**p,'slog':lambda a,p:np.sign(a)*np.log1p(abs(a)),'pow':lambda a,p:np.sign(a)*abs(a)**p,'gauss':lambda a,p:np.exp(-a*a),'softplus':lambda a,p:np.logaddexp(0,a),'cos':lambda a,p:np.cos(a)}
I,J=np.triu_indices(4,1); K,L=np.triu_indices(4); tri=list(itertools.combinations(range(4),3)); folds=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.r_[0:24,44:72],np.arange(24,44))]
def bank(z,h,m):
 q=h[:,K]*h[:,L]; qq=z[:,K]*z[:,L]; c=np.column_stack([np.prod(h[:,i],axis=1) for i in tri]); cross=h[:,I]*h[:,J]
 return {'diag':lambda:np.c_[aff(h),h*h], 'quad':lambda:np.c_[aff(h),q], 'triple':lambda:np.c_[aff(h),q,c], 'raw':lambda:np.c_[aff(h),z], 'rawquad':lambda:np.c_[aff(h),z,q], 'rawpoly':lambda:np.c_[aff(h),z,qq], 'mixed':lambda:np.c_[aff(h),z,q,z*h], 'cross':lambda:np.c_[aff(h),cross], 'lin':lambda:aff(h), 'cubdiag':lambda:np.c_[aff(h),q,h**3]}[m]()
def fit(fn,D,z,v,ix,which,lam=.008):
 extra=which in ['hyp','rat','soft','pow']; p0=([.7,.6] if extra else [.7]); lo=([-1.7,.13] if extra else [-1.7]); hi=([2.,2.8] if extra else [2.]);
 def eval(p):return D(z,fn(np.exp(p[0])*z+(p[len(p0):] if len(p)>len(p0) else 0),p[1] if extra else 1.))
 def profile(A):
  X=A[ix]; norm=np.maximum(np.sqrt(np.mean(X*X,axis=0)),.001); reg=lam*norm; reg[0]=0
  w=np.linalg.solve(X.T@X+np.diag(reg**2+1e-12),X.T@v[ix]);return w,reg
 def res(p):
  A=eval(p); w,reg=profile(A); e=A[ix]@w-v[ix]; return np.r_[e.ravel(),(reg.reshape((-1,)+(1,)*(w.ndim-1))*w).ravel(),.025*p[len(p0):]]
 results=[]
 for start in (p0, [0.15]+p0[1:]):
  f=least_squares(res,start, bounds=(lo,hi),max_nfev=35); results.append(f)
 f=min(results,key=lambda a:a.cost); A=eval(f.x); w,_=profile(A); return A@w,f.x,np.mean(abs((A@w-v)[ix]))
if __name__=='__main__':
 for ky in ('v1','v2'):
  P=s['panel_c']; u,v=P['u'],P[ky]; scores=[]
  for name,fn in F.items():
   for m in ('raw','rawpoly','diag','quad'):
    d=lambda z,h: bank(z,h,m); pred,pars,err=fit(fn,d,u,v,np.arange(len(u)),name,0.0001); scores.append((err,name,m,pars))
  print('panel',ky,sorted(scores,key=lambda a:a[0])[:8])
 for z,zn in [(b,'b'),(x,'x')]:
  scores=[]
  for name,fn in F.items():
   for m in ('diag','quad','triple','raw','rawquad','rawpoly','mixed','cross','cubdiag'):
    d=lambda z,h: bank(z,h,m); vv=[]
    for i,j in folds:
     pred,p,e=fit(fn,d,z,y,i,name); vv.append((np.mean(abs(pred[j]-y[j])),e,p.round(4)))
    scores.append((np.mean([a[0] for a in vv]),name,m,vv))
  print('scalar',zn,flush=True)
  for score in sorted(scores,key=lambda a:a[0])[:15]:print(score,flush=True)
