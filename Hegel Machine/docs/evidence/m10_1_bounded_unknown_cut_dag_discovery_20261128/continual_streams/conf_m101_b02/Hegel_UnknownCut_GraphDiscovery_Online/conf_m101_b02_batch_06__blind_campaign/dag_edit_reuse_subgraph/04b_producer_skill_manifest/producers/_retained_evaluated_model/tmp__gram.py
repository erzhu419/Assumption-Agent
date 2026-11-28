from probe import *
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist

def one(x): return np.c_[np.ones(len(x)),x]
def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def p2(x): return np.c_[x,x*x]
def pair(x): return x*np.roll(x,-1,axis=1)
w=ls(one(pair(u)),v1); w2=ls(one(p2(u)),v2)
def c(x): return one(pair(x))@w
def d(x): return one(p2(x))@w2
x=s['train']['x']; n=len(x); trn=np.arange(54); val=np.arange(54,n); folds=np.array_split(np.random.default_rng(781).permutation(n),4)
Y=np.c_[y,tr]; T=np.linalg.svd(tr,full_matrices=False)[2][:3].T
print('raw ranges',x.min(0),x.max(0),'z-x correlations',np.corrcoef(np.c_[z,x].T)[:4,4:]);print('trajectory coefficients by z and c(b)',np.round(ls(one(np.c_[z,c(b)]),tr),3))
def ev(pred,idx):
 r=np.abs(pred-Y[idx]);return np.array([r[:,0].mean(),r[:,1:].mean()])
def cv_m(M):
 res=[]
 for idx in [val,*folds]:
  a=np.setdiff1d(np.arange(n),idx);res.append(ev(M[idx]@ls(M[a],Y[a]),idx))
 return np.round([res[0],np.max(res[1:],axis=0)],5)
base=np.c_[z,b,c(b),d(b)]; funcs={'id':lambda q:q,'atan':np.arctan,'sin':np.sin,'tanh':np.tanh,'signsqrt':lambda q:np.sign(q)*np.sqrt(abs(q))}
sets={'x':x,'z':z,'b':b,'cz':c(z),'cb':c(b),'db':d(b),'xzb':np.c_[x,z,b],'zbc':base,'zbcz':np.c_[z,b,c(z),c(b),d(z)],'cascade':np.c_[base,c(d(b)),d(c(b)),c(c(b))]}
for name,F in sets.items():
 print('lin',name,cv_m(one(F)))
 if F.shape[1]==4:
  for lab,g in funcs.items():
   G=g(F)
   for deg in (2,3,4):
    M=one(np.column_stack([np.prod(G[:,v],axis=1) for j in range(1,deg+1) for v in combinations_with_replacement(range(4),j)]));print('poly',name,lab,deg,cv_m(M))
for name,F in sets.items():
 F=(F-F.mean(0))/np.maximum(F.std(0),1e-8); D=cdist(F,F,'sqeuclidean')/F.shape[1]; L=1+F@F.T/F.shape[1]; opts=[]
 for typ in ('rbf','matern','rq','poly'):
  for q in (.15,.5,1.5,4.):
   A=np.sqrt(D*q); K={'rbf':np.exp(-D*q),'matern':(1+A+A*A/3)*np.exp(-A),'rq':1/(1+D*q),'poly':L**int(q+2)}[typ]+.2*L
   for reg in (1e-5,1e-3,.02):
    es=[]
    for idx in [val,*folds]:
     a=np.setdiff1d(np.arange(n),idx);es.append(ev(K[np.ix_(idx,a)]@np.linalg.solve(K[np.ix_(a,a)]+reg*np.eye(len(a)),Y[a]),idx))
    opts.append((typ,q,reg,np.round(es[0],5),np.round(np.max(es[1:],axis=0),5)))
 for j in (0,1): print('kernel',name,j,sorted(opts,key=lambda r:r[4][j])[:1])
