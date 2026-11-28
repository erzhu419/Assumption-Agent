import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/graph.py').read())
from scipy.optimize import least_squares
F={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'erf':__import__('scipy.special',fromlist=['erf']).erf,'asinh':np.arcsinh,'softsign':lambda a:a/np.sqrt(1+a*a),'softplus':lambda a:np.logaddexp(0,a),'silu':lambda a:a/(1+np.exp(-a)),'gelu':lambda a:a*__import__('scipy.special',fromlist=['ndtr']).ndtr(a)}
ops2=[]
for V in (R,S):
 results=[]
 for nm,f in F.items():
  for mode in ('pure','plus'):
   def feat(W,p):
    H=f(W*p[:4]+p[4:]);return aff(H if mode=='pure' else np.c_[W,H])
   def solve(idx):
    def res(p):
     E=feat(U[idx],p);return (E@fit(E,V[idx])-V[idx]).ravel()
    o=min((least_squares(res,np.r_[np.full(4,k),np.zeros(4)],bounds=([.15]*4+[-2]*4,[5]*4+[2]*4),max_nfev=100) for k in (.8,1.8,2.8)),key=lambda o:np.linalg.norm(o.fun)); p=o.x;w=fit(feat(U[idx],p),V[idx]);return p,w
   p,w=solve(np.arange(18));e=np.mean(abs(feat(U[18:],p)@w-V[18:]));results.append((e,nm,mode,p))
 print('panel link candidates',sorted(results,key=lambda q:q[0])[:6]); _,nm,mode,_=min(results,key=lambda q:q[0]);f=F[nm];p,w=solve(np.arange(24)); print('full panel',nm,mode,p,w,np.std(feat(U,p)@w-V));ops2.append((nm,mode,p,w))
def app(W,op):
 nm,mode,p,w=op;H=F[nm](W*p[:4]+p[4:]);return aff(H if mode=='pure' else np.c_[W,H])@w
# Look for small predictive cut sets using cross-validated quadratic libraries.
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from sklearn.model_selection import KFold
ss={}
for name,W in [('z',z),('Bz',z@B+b),('invBz',(z-b)@np.linalg.inv(B)),('x',x)]:
 H,J=[app(W,op) for op in ops2];ss[name]=W;ss[name+'1']=H;ss[name+'2']=J;ss[name+'12']=np.c_[H,J];ss[name+'all']=np.c_[W,H,J]
K=[]
for name,W in ss.items():
 W=(W-W.mean(0))/W.std(0);d=cdist(W,W,'sqeuclidean')/W.shape[1]; p=W@W.T/W.shape[1]
 for typ,M in [('rbf',np.exp(-d/2)),('rbfl',np.exp(-d/8)),('rbfs',np.exp(-d*2)),('p3',(1+p)**3),('p4',(1+p)**4),('p5',(1+p)**5)]:
  K.append((name,typ,M))
K += [('mix',typ,np.mean([M for name,t,M in K if t==typ and name.startswith('z')],axis=0)) for typ in ('rbf','rbfl','rbfs','p3','p4','p5')]
folds=list(KFold(6,shuffle=True,random_state=932).split(z))
res=[]
for name,typ,M in K:
 for lam in (.00001,.0001,.001,.01):
  pp=np.empty_like(Y)
  for i,j in folds:
   pp[j]=M[np.ix_(j,i)]@solve(M[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i],assume_a='pos')
  E=np.c_[abs(pp[:,2]-y),abs(pp[:,:2]@D.T-T)];res.append((E[:,0].mean(),E[:,1:].mean(0).max(),name,typ,lam))
print('best scalar',sorted(res)[:12]);print('best trajectory',sorted(res,key=lambda q:q[1])[:12]);print('stats',T.mean(0),T.std(0),y.mean(),y.std(),C.mean(0),C.std(0),'summary',tr['summary'])
