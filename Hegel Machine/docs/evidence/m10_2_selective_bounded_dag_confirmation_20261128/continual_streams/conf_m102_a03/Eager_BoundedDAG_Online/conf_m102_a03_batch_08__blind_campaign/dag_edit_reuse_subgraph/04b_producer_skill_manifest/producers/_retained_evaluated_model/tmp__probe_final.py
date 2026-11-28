exec(open('/tmp/probe_panels.py').read().split('for k,panel')[0])
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
B=s['panel_b']; C=s['panel_c']; Y=np.c_[y,T]; fs=[]; N=len(y); I=np.arange(N)
def pc(q):return [(q[:,i]*q[:,j]) for i in range(q.shape[1]) for j in range(i+1,q.shape[1])]
def inv(x,k):
 q=x.copy()
 for i in range(30):q-=(q+k*q**3-x)/(1+3*k*q*q)
 return q
def mapping(x,v):
 funs={'sin':(lambda x,k: np.sin(x*k),3.),'tanh':(lambda x,k: np.tanh(x*k),1.4),'rat':(lambda x,k:x/(1+k*x*x),.8),'inv':(inv,2.)}
 out={}
 for nm,(f,k0) in funs.items():
  for mode in (0,1,2):
   def d(q,k):
    w=f(q,k);return np.c_[np.ones(len(q)),q,w] if mode==0 else (np.c_[np.ones(len(q)),q,q*q,w] if mode==1 else np.c_[np.ones(len(q)),w])
   def resid(k):
    X=d(x,k);return (X@reg(X,v)-v).ravel()
   opt=least_squares(resid,np.full(4,k0),bounds=(.03,12.),max_nfev=120)
   X=d(x,opt.x); H=X@np.linalg.pinv(X); E=v-H@v; loo=np.mean(abs(E/np.maximum(1e-7,1-H.diagonal())[:,None]));out[nm+str(mode)]=(loo,d(u,opt.x)@reg(X,v));print(nm,mode,opt.x,loo,np.mean(abs(E)))
 return out
print('B',flush=True); b=mapping(B['u'],B['v']); print('C1',flush=True); c=mapping(C['u'],C['v1']);
for nm,(e,w) in sorted(b.items(),key=lambda q:q[1][0])[:6]: fs.append(('B'+nm,w))
for nm,(e,w) in sorted(c.items(),key=lambda q:q[1][0])[:3]: fs.append(('C'+nm,w))
fs.append(('u',u)); folds=[(tr,te)]+[(np.setdiff1d(I,j),j) for j in np.array_split(np.random.default_rng(133).permutation(N),6)]; results=[]
def test(nm,X,K=None):
 if K is None:
  X=X/np.maximum(X.std(0),.001); X[:,0]=1; K=X@X.T
 for l in (1e-6,.001,.01,.05,.2,1.):
  ee=[]
  for a,b in folds:
   r=abs(K[np.ix_(b,a)]@np.linalg.solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a])-Y[b]);ee.append([r[:,0].mean(),r[:,1:].mean()]);
  results.append((nm,l,ee[0],np.mean(ee[1:],0),np.max(ee[1:],0)))
for nm,w in fs:
 for kn,q in [('w',w),('u',u),('inv',inv(u,2)),('uw',np.c_[u,w])]:
  for j in (1,2,3):
   for deg in (1,2,3,5,7):
    if nm=='u' and kn in ('w','uw'):continue
    X=np.c_[poly(w,j),*[q**i for i in range(1,deg+1)]];test((nm,kn,j,deg),X)
for scale in (.1,.25,.45,.7,1.,1.5,2.,3.):
 for nm,w in fs:
  for tag,q in [('w',w),('uw',np.c_[u,w])]:
   v=q/np.maximum(.1,q.std(0)); D=cdist(v,v,'sqeuclidean'); K=np.exp(-D*scale); K2=np.mean(np.exp(-scale*(v[:,None,:]-v[None,:,:])**2),2); P=poly(w,2); P=P/np.maximum(P.std(0),.02);P[:,0]=1
   for a in (0,.2,1,5):test((nm,tag,scale,a),None,K+a*P@P.T+K2)
for k in (0,1):
 print('top tail',k,*sorted(results,key=lambda r:r[2][k])[:8],sep='\n');print('top combined',k,*sorted(results,key=lambda r:r[2][k]+r[3][k])[:8],sep='\n')
