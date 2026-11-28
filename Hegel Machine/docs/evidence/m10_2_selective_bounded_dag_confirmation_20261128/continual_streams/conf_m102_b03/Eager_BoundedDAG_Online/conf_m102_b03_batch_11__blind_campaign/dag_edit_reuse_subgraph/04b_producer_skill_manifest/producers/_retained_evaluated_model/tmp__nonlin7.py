exec(open('/tmp/panel_search.py').read().split('C={')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import qr
from scipy.special import expit
from sklearn.gaussian_process.kernels import Matern
print('sv',np.linalg.svd(t,compute_uv=False)); print('ranges',[(k,np.min(s[k]['u'],0),np.max(s[k]['u'],0)) for k in ('panel_a','panel_b','panel_c')]); print('summary',s['train']['summary'])
X=np.c_[y,t]; rng=np.random.default_rng(3128); folds=np.array_split(rng.permutation(len(y)),4)
def cv(K,F=None):
 err=[]
 for j in folds:
  i=np.setdiff1d(np.arange(len(y)),j); L=K[np.ix_(i,i)]; R=K[np.ix_(j,i)]
  if F is None: pred=R@np.linalg.solve(L,X[i])
  else:
   f=F[i]; V=np.linalg.solve(L,np.c_[X[i],f]); p=ls(np.c_[f.T@V[:,X.shape[1]:]],f.T@V[:,:X.shape[1]]);pred=R@(V[:,:X.shape[1]]-V[:,X.shape[1]:]@p)+F[j]@p
  err.append(abs(pred-X[j]))
 E=np.concatenate(err);return E[:,0].mean(),E[:,1:].mean()
res=[]
for wname,w in [('z',z),('x',x),('f',z@B+h),('r',(z-h)@P)]:
 c,d=(f(w) for f in funcs); dp=(c*d).sum(1); base=ones(np.c_[c,d,dp]); cross=(c[:,:,None]*d[:,None,:]).reshape(len(w),-1)
 for nm,U in [('z',w),('cd',np.c_[c,d]),('zcd',np.c_[w,c,d]),('weightedcd',np.c_[c,0.5*d])]:
  U=U/U.std(0); dist=cdist(U,U)
  for l in [.75,1.5,3,6,12,24]:
   v=dist/l
   for typ,K in [('rbf',np.exp(-.5*v*v)),('matern',(1+v+v*v/3)*np.exp(-v))]:
    for ridge in [1e-7,1e-4,.003,.03]:
     for fn,F in [('simple',ones(w)),('branch',base),('cross',np.c_[base,cross])]:
      e=cv(K+np.eye(len(y))*ridge,F);res.append((*e,wname,nm,l,typ,ridge,fn))
print('GP y',sorted(res)[:9]);print('GP t',sorted(res,key=lambda x:x[1])[:9]);
# Single nonlinear index: jointly fitted direction, partially linear readout.
for wname,w in [('z',z),('x',x),('f',z@B+h),('r',(z-h)@P)]:
 c,d=(f(w) for f in funcs); dot=(c*d).sum(1); base=ones(np.c_[c,d,dot]);
 for name,g in [('tanh',np.tanh),('sin',np.sin),('soft',lambda u:u/(1+abs(u))),('gauss',lambda u:np.exp(-u*u)),('rat',lambda u:u/(1+u*u)),('relu',lambda u:np.logaddexp(0,u))]:
  for mod in [False,True]:
   vals=[]
   for j in folds:
    i=np.setdiff1d(np.arange(len(y)),j)
    def fd(p):
     v=g(w@p[:4]+p[4]);return np.c_[base,v[:,None]*np.c_[np.ones(len(y)),w]] if mod else np.c_[base,v]
    def loss(p):
     f=fd(p)[i];return f@ls(f,y[i])-y[i]
    ps=[least_squares(loss,p,max_nfev=60) for p in np.r_[np.eye(5)*1.5,rng.normal(0,1,(5,5))]]; p=min(ps,key=lambda q:q.cost).x; f=fd(p); vals.append([np.mean(abs(f[j]@ls(f[i],y[i])-y[j])),np.sqrt(np.mean(loss(p)**2))]);
   print(wname,name,mod,np.mean(vals,0))
