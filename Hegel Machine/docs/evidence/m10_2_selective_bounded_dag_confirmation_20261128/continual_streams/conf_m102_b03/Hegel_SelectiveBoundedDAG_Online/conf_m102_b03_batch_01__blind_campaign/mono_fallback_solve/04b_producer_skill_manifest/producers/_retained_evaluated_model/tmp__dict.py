exec(open('/tmp/dag.py').read().split('ops=')[0])
from itertools import combinations, product
from scipy.sparse.linalg import svds
N=52
W={'z':z,'L':L(z),'Q':Q(z),'R':R(z),'L0':z@B[1:]}
Z=np.c_[y,ts@np.linalg.pinv(H.T)]
Dct={}
def add(name,w):
 w=np.asarray(w).reshape(len(z),-1)
 for j,v in enumerate(w.T):
  if v.std()>1e-8:Dct[f'{name}[{j}]']=v
for n,w in W.items():
 add(n,w);add(n+'^2',w*w);add(n+'^3',w**3)
 for p in [.625,1.25,1.8,2.4,3.6,4.8,6.0]:
  for name,fn in [('s',np.sin),('c',np.cos),('t',np.tanh),('r', lambda a:1/(1+a*a))]:add(f'{name}{p}({n})',fn(w*p))
 for k,l in combinations(range(4),2):
  add(f'{n}{k}*{l}',w[:,k]*w[:,l])
  for a in [-1,1,-.5,.5,-2,2]:
   v=w[:,k]+a*w[:,l]
   for p in [1.25,2.4,3.6,4.8]:
    for name,fn in [('s',np.sin),('c',np.cos),('t',np.tanh),('r',lambda a:1/(1+a*a))]: add(f'{name}{p}({n}{k}+{a}*{l})',fn(p*v))
 for ids in combinations(range(4),3):add(f'{n}prod{ids}',w[:,ids].prod(1))
for n,m in combinations(W,2):
 w,v=W[n],W[m]; prod=(w[:,:,None]*v[:,None,:]).reshape(len(w),-1);add(n+'*'+m,prod);add(n+'dot'+m,np.sum(w*v,1));
 for a in [.22,.5,1.25,2.4]:
  add(f'{n}/(1+{a}{m}^2)',(w[:,:,None]/(1+a*v[:,None,:]**2)).reshape(len(w),-1));add(f'{n}*tanh({a}{m})',(w[:,:,None]*np.tanh(a*v[:,None,:])).reshape(len(w),-1))
for n in ['z','L','Q','R']:
 w=W[n]
 for j in range(4):
  for th in [-.4,0,.4]:
   for a in [1.25,2.4,5,10,30]:
    g=np.tanh(a*(w[:,j]-th))
    for m in ['z','L','Q','R']:
     v=W[m];add(f'{m}*g({n}{j},{th},{a})',v*g[:,None]);
names=list(Dct);F=np.array(list(Dct.values())).T; F=(F-F[:N].mean(0))/F[:N].std(0); F0=F[:N];
print('dict',F.shape)
for j in range(4):
 Y=Z[:,j]; sel=[]; residual=Y[:N]-Y[:N].mean();basis=np.ones((N,1))/np.sqrt(N); best=[]
 for k in range(13):
  P=F0-basis@(basis.T@F0); den=np.linalg.norm(P,axis=0);scores=abs(residual@P)/np.maximum(den,1e-8); ix=np.argmax(scores); sel.append(ix);basis=np.c_[basis,P[:,ix]/den[ix]]; X=ones(F[:,sel]); e=Y-X@ls(X[:N],Y[:N]); residual=e[:N]; best.append((names[ix],np.mean(abs(e[:N])),np.mean(abs(e[N:]))))
 print(j,best)
# Group nonlinear fits: pairwise products, orthogonal branch interactions and saturated projections.
for j in range(4):
 Y=Z[:,j]; print('leading single',j,[(names[k],round((Y[:N]-Y[:N].mean())@F0[:,k]/N,4)) for k in np.argsort(abs((Y[:N]-Y[:N].mean())@F0))[-12:][::-1]])
