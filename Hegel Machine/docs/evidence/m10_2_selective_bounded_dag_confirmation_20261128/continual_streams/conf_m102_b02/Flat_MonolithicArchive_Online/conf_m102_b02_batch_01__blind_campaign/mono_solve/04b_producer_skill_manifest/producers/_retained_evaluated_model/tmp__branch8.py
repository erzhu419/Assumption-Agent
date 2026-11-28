exec(open('/tmp/graph7.py').read().split('print(\'summary\'')[0])
from scipy.optimize import minimize_scalar
from itertools import combinations
rng=n.random.default_rng(33)
# Probe residual updates, including auxiliary operators evaluated at the same intermediate input.
B,C=v,v1
rv=n.linalg.svd(T[:52],full_matrices=False)[2][:2]; target=n.c_[y,T@rv.T]; sc=target[:52].std(0)
splits=[n.arange(72)]+[rng.permutation(72) for _ in range(4)]
def reg(D,a):
 mu=D.mean(0); sig=n.maximum(.005,D.std(0)); return n.c_[n.ones(len(D)),(D-mu)/sig]
def assess(D,allfold=False):
 es=[]
 for P in (splits if allfold else splits[:1]):
  i,j=P[:52],P[52:];a=D[i]; d,U=n.linalg.eigh(a@a.T); pred=(D[j]@a.T@U)[None,:,:]@((U.T@target[i])[None,:,:]/(d[None,:,None]+n.array([.00001,.0002,.003,.03,.2])[:,None,None])); es.append(n.mean(abs(pred-target[j]),axis=1))
 return n.array(es)
def features(W,typ):
 if typ=='lin':return reg(n.c_[tuple(W)],0)
 if typ=='diag':return reg(n.c_[tuple(W+[w*w for w in W])],0)
 if typ=='cross':return reg(n.c_[tuple(W+[W[i]*W[j] for i,j in combinations(range(len(W)),2)])],0)
 if typ=='full':return reg(n.c_[tuple([poly(w,2)[:,1:] for w in W])],0)
 if typ=='cube':return reg(n.c_[tuple([poly(w,2)[:,1:] for w in W]+[w**3 for w in W])],0)
rows=[]
for a in [-2,-1,-.5,-.22,0,.22,.5,.75,1,1.25,1.5,2,3,5]:
 for init,w in [('z+aB',z+a*B(z)),('z+aC',z+a*C(z)),('B+az',B(z)+a*z),('z+aV',z+a*V(z)),('C+az',C(z)+a*z),('aB',a*B(z)),('aC',a*C(z))]:
  W={'z':z,'w':w,'B':B(w),'C':C(w),'V':V(w),'BC':B(C(w)),'CB':C(B(w)),'VB':V(B(w)),'VC':V(C(w))}
  groups=[k for m in [1,2,3] for k in combinations(W,m) if m<3 or k[0] in ['z','w']]+[('w','B','C','V'),('w','C','V','CB'),('z','C','V','CB'),('w','C','V','VB')]
  for keys in groups:
   for typ in ['lin','diag','cross','full','cube']:
    D=features([W[k] for k in keys],typ); e=assess(D)[0];
    for col in range(3):
     i=e[:,col].argmin();rows.append((e[i,col],col,a,init,keys,typ,i,e[i],D)) if e[i,col]/sc[col]<.19 else None
for col in range(3):
 print('BEST',col)
 for _,_,a,init,keys,typ,i,e,D in sorted((r for r in rows if r[1]==col),key=lambda r:r[0])[:12]:print(a,init,keys,typ,i,e,'CV',assess(D,True)[1:,i,col])
