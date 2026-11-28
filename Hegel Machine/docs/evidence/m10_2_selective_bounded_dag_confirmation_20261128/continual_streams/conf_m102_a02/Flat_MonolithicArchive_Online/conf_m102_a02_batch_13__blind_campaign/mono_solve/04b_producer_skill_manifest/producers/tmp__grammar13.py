exec(open('/tmp/graph13.py').read().split('h=tr[')[0])
from sklearn.model_selection import KFold
from scipy.linalg import cho_solve,cho_factor
b=bs['cosx'](x); y=tr['y']; C=c1(b); D=c2(b); F=quad(b); A=aff(b)
folds=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=41).split(b)]; print('b mean/range',b.mean(0),b.min(0),b.max(0));
def cv(F):return [np.mean(abs(F[j]@ls(F[i],y[i])-y[j])) for i,j in folds]
R=[]
for nm,z in [('b',b),('cosx',np.cos(2.6*x)),('C',C),('D',D)]:
 p,q=np.triu_indices(4,1); U=z[:,p];V=z[:,q]; M={ 'z': z,'prod':U*V,'sum':U+V,'diff':U-V,'cube':z**3,'tri':np.c_[[np.prod(z[:,j],axis=1) for j in itertools.combinations(range(4),3)]].T};
 for gn,g in M.items():
  for j,w in enumerate(g.T):
   for k in (.22,.5,1,1.25,2.15,2.6):
    for fn,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('r',lambda x:1/(1+x*x)),('rr',lambda x:x/(1+x*x))]:
     G=f(k*w); e=cv(np.c_[F,G]);R.append((np.mean(e[1:]),e[0],max(e[1:]),nm,gn,j,fn,k))
  print('BLOCK',nm,gn,cv(np.c_[F,g]))
print('single additions',*sorted(R)[:24],sep='\n');
# Kernel overrides with an unpenalized affine/quadratic mean.
for nm,z in [('b',b),('cos',np.cos(2.6*x)),('D',D),('C',C)]:
 for std in [False,True]:
  X=(z-z.mean(0))/(z.std(0) if std else 1); T=np.sum((X[:,None]-X[None,:])**2,2); H=X@X.T/4
  for label,K in [(f'rbf{a}',np.exp(-a*T)) for a in (.025,.07,.15,.4,1)]+[(f'poly{a}',(1+H)**a) for a in (3,4,5,6,8)]:
   for M in [A,F]:
    for reg in (1e-6,1e-4,.001,.01):
     es=[]
     for i,j in folds:
      W=cho_factor(K[np.ix_(i,i)]+reg*np.eye(len(i))); wi=cho_solve(W,M[i]); c=ls(M[i].T@wi,wi.T@y[i]); p=M[j]@c+K[np.ix_(j,i)]@cho_solve(W,y[i]-M[i]@c);es.append(np.mean(abs(p-y[j])))
     R.append((np.mean(es[1:]),es[0],max(es[1:]),nm,std,label,M.shape[1],reg))
print('all',*sorted(R,key=lambda r:(max(r[0],r[1]),r[2]))[:30],sep='\n')
