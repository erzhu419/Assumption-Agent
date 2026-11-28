import pickle,itertools as it,numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; z=d['traj']; U,S,V=np.linalg.svd(z,full_matrices=False); Y=np.c_[d['y'],U[:,:2]*S[:2]]; n=len(x); np.set_printoptions(precision=4,suppress=True)
def a(x):return np.c_[np.ones(len(x)),x]
p=s['panel_a']; mask=np.ones(len(p['u']),bool)
for _ in range(4):
 A=np.linalg.lstsq(a(p['u'])[mask],p['v'][mask],rcond=None)[0];r=np.linalg.norm(a(p['u'])@A-p['v'],axis=1); mask=r<4*np.median(r)
p=s['panel_b']; B=np.linalg.lstsq(a(p['u']),p['v'],rcond=None)[0];print('B orthogonality',B[1:]@B[1:].T)
xa=(x-A[0])/np.diag(A[1:]); xb=a(x)@B; xab=a(xa)@B; xbi=(x-B[0])@np.linalg.inv(B[1:]); xabi=(xa-B[0])@np.linalg.inv(B[1:]); xba=(xb-A[0])/np.diag(A[1:]); bases={'raw':x,'Ainv':xa,'B':xb,'AinvB':xab,'Binv':xbi,'AinvBinv':xabi,'BAinv':xba}
folds=[(np.arange(56),np.arange(56,72))]; rng=np.random.default_rng(22); perm=rng.permutation(n); folds += [(np.setdiff1d(np.arange(n),j), j) for j in np.array_split(perm,5)]
def score(K,lam):
 return np.array([np.mean(abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]),0) for i,j in folds])
for name,q in bases.items():
 q=q/np.std(q,axis=0); P=np.c_[a(q),*[q[:,i]*q[:,j] for i,j in it.combinations_with_replacement(range(4),2)]]; K=P@P.T; dd=cdist(q,q,'sqeuclidean'); results=[]
 def add(label,E):
  for l in [1e-5,1e-4,1e-3,.01]:
   r=score(K+E,l);results.append((label,l,r[0],r[1:].mean(0),r[1:].max(0)))
 for h in (1.,1.7,2.6,4.):
  add(('rbf',h),np.exp(-dd/(2*h*h)))
  for f in (.7,1.2,1.7,2.3):
   G=np.c_[np.sin(q*f),np.cos(q*f),np.tanh(q*f)];add(('additive',f,h),G@G.T+.05*np.exp(-dd/(2*h*h)))
 for degree in (3,4):
  G=np.column_stack([np.prod(q[:,ix],axis=1) for ix in it.combinations_with_replacement(range(4),degree)]);add(('poly',degree),.1*G@G.T)
 print(name,'best by output (scalar / amplitude combined):');
 for k in (0,1):
  for row in sorted(results,key=lambda r:r[3][0] if k==0 else np.mean(r[3][1:]))[:2]:print(row)
# Inspect which rotation makes quadratic curvature compact.
for name in ('raw','AinvB','AinvBinv'):
 q=bases[name]; pairs=list(it.combinations_with_replacement(range(4),2)); P=np.c_[a(q),*[q[:,i]*q[:,j] for i,j in pairs]]; C=np.linalg.lstsq(P,Y,rcond=None)[0];print(name,'linear',C[:5].T,'quad',C[5:].T)
