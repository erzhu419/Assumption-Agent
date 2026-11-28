exec(open('/tmp/graph7.py').read().split("print('summary'")[0])
from sklearn.linear_model import Ridge
from itertools import combinations
B,C=v,v1
P={'z':z,'C':C(z),'CB':C(B(z)),'B':B(z),'V':V(z),'VB':V(B(z))}; F=temporal(n.log([.41,1.795])); H=ls(F,T.T).T; r=n.linalg.svd(T[:52],full_matrices=False)[2][:2]; Q=n.c_[y,T@r.T,H]; rng=n.random.default_rng(218); splits=[n.arange(72)]+[rng.permutation(72) for _ in range(18)];
def design(g,kind):
 w=[P[k] for k in g]; keys=[f'{k}{j}' for k in g for j in range(4)]; D=list(n.column_stack(w).T); names=keys.copy()
 for i,k in enumerate(g):
  for j in range(i,len(g)):
   if i==j and kind in ['no_sq','inter']:continue
   for a in range(4):
    for b in range(4):
     if (kind!='full' and a!=b) or (i==j and a>b):continue
     D.append(w[i][:,a]*w[j][:,b]); names.append(f'{k}{a}*{g[j]}{b}')
 return n.column_stack(D),names
def cv(D,alpha):
 R=[]
 for s in splits:
  i,j=s[:52],s[52:]; m=D[i].mean(0); sd=n.maximum(D[i].std(0),1e-8); a=(D[i]-m)/sd; q=(D[j]-m)/sd; y0=Q[i].mean(0); c=n.linalg.solve(a.T@a+alpha*n.eye(a.shape[1]),a.T@(Q[i]-y0)); pred=q@c+y0; et=abs(pred[:,1:3]@r-T[j]); R.append(n.r_[n.mean(abs(pred-Q[j]),0),et.mean(),et.mean(0)[-1]])
 return n.array(R)
rows=[]
for g in [('z','CB'),('z','C','CB'),('z','C','CB','B'),('z','C','CB','V'),('z','C','CB','VB')]:
 for kind in ['full','diag','no_sq']:
  D,names=design(g,kind)
  for a in [.0003,.003,.03,.1,.3]:
   e=cv(D,a);rows.append((g,kind,a,e[0],e.max(0),e.mean(0)))
for j in [0,5,6]:
 print('BEST',j)
 for g,k,a,e,m,o in sorted(rows,key=lambda row:row[4][j])[:6]: print(g,k,a,'public',e,'worst',m,'mean',o)
for g in [('z','CB'),('z','C','CB')]:
 D,keys=design(g,'diag'); A=n.c_[n.ones(72),D]; d=n.r_[1,D.std(0)]; c=ls(A[:52]/d,Q[:52]); print('COEF',g);print('RMS',n.std(A@ (c/d[:,None])-Q,0));print(*[(key,n.round(w,4)) for key,w in zip(['bias']+keys,c/d[:,None])],sep='\n')
