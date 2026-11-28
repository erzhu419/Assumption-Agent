exec(open('/tmp/mixed07.py').read().split('acts=')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations, combinations_with_replacement
n=len(y); Y=np.c_[y,T]; sc=Y.std(0); I=np.eye(n); idx=np.arange(n); folds=[np.arange(52,72)]+[np.arange(j,j+18) for j in range(0,72,18)]
def feat(x,k): return np.column_stack([np.prod(x[:,i],axis=1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def norm(x):return (x-x.mean(0))/np.maximum(x.std(0),1e-4)
def gram(x): x=norm(x); return (x@x.T)/x.shape[1]
def cv(K,alpha):
 es=[]
 for h in folds:
  tr=np.setdiff1d(idx,h); pred=K[np.ix_(h,tr)]@solve(K[np.ix_(tr,tr)]+alpha*I[np.ix_(tr,tr)],Y[tr],assume_a='pos'); E=abs(pred-Y[h]);es.append([E[:,0].mean(),E[:,1:].mean()])
 return np.r_[np.max(es,axis=0),es[0]]
a=g._aff(np.c_[v,pair(v),q]); K0=1+gram(a[:,1:]); sources={'U':u,'V':v,'Q':q,'PQ':np.c_[pair(v),q],'VPQ':a[:,1:],'UVQ':np.c_[u,v,q]}; rows=[]
for name,x in sources.items():
 L=gram(x); D=cdist(norm(x),norm(x),'sqeuclidean')/x.shape[1]
 for kind,K in [('lin',L),('quad',(1+L)**2),('cube',(1+L)**3),('q3',gram(feat(x,3)))]+[(f'rbf{r}',np.exp(-r*D)) for r in [.08,.25,.7,1.5]]+[(f'mat{r}',(1+np.sqrt(r*D))*np.exp(-np.sqrt(r*D))) for r in [.5,2,5]]:
  for weight in [.05,.3,1.,3.,20.]:
   for alpha in [.00001,.0003,.003,.015,.06,.2]: rows.append((cv(K0+weight*K,alpha),name,kind,weight,alpha))
for ix in [0,1,2,3]: print('KERNEL',ix,*sorted(rows,key=lambda r:r[0][ix])[:4],sep='\n')
# Explicit analogues of a gated panel-C mixture and triples in calibrated space.
from scipy.optimize import least_squares
for Xname,X0 in [('vq',np.c_[v,q]),('vpq',a[:,1:]),('uvq',np.c_[u,v,q])]:
 X=g._aff(norm(X0)); A=a.copy(); A[:,1:]=norm(A[:,1:]); M=A.shape[1]; Yn=Y/sc
 for mode in [0,1,2]:
  es=[]
  for h in folds:
   tr=np.setdiff1d(idx,h); Atr=A[tr]; Xtr=X[tr]; yt=Yn[tr]; L=np.linalg.lstsq(Atr,yt,rcond=None)[0]; R=yt-Atr@L; uu,ss,vv=np.linalg.svd(R,full_matrices=False); direction=vv[0]; rr=R@direction; rng=np.random.default_rng(17); best=None
   def fun(Z,W,p):
    t=Z@p[:Z.shape[1]]; k=p[Z.shape[1]:Z.shape[1]+M]; m=p[-M:];
    if mode==0: f=np.tanh(t)
    elif mode==1: f=1/(1+t*t)
    else:f=np.sin(t)
    return W@k+f*(W@m)
   for it in range(16):
    start=np.r_[rng.normal(0,.65,X.shape[1]),np.zeros(M),rng.normal(0,.15,M)]
    fit0=least_squares(lambda p:np.r_[fun(Xtr,Atr,p)-rr,.045*p],start,max_nfev=100); val=np.dot(fit0.fun,fit0.fun)
    if best is None or val<best[0]:best=val,fit0.x
   pred=(A[h]@L+fun(X[h],A[h],best[1])[:,None]*direction)*sc; e=abs(pred-Y[h]); es.append([e[:,0].mean(),e[:,1:].mean()])
  print('GATE',Xname,mode,np.max(es,0),es[0],flush=True)
