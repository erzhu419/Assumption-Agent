exec(open('/tmp/operators.py').read().split('ops={}')[0])
from scipy.special import erf
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations
p=s['panel_b']; u,v=p['u'],p['v']; print('ranges',np.ptp(u,axis=0),np.ptp(z,axis=0)); print('times',t,'traj spectrum',np.linalg.svd(Y[:,1:],compute_uv=False)); rng=np.random.default_rng(41); folds=np.array_split(rng.permutation(len(u)),6)
def models(X,w,kind):
 h=w[0]*X; sh=np.sin(h); cols=[lin(X)];
 if kind.startswith('sin'): cols+=[sh]
 if kind.startswith('tanh'): cols+=[np.tanh(h)]
 if kind.startswith('erf'): cols+=[erf(h)]
 if kind.endswith('diag2'): cols+=[X**2]
 if kind.endswith('even'): cols+=[np.cos(h)]
 if kind.endswith('quad'): cols+=[poly(X,2)[:,5:]]
 if kind.endswith('cross'): cols+=[np.column_stack([X[:,i]*X[:,j] for i,j in combinations(range(4),2)])]
 if kind.endswith('rad'):cols+=[X*(X**2).sum(1)[:,None]]
 return np.column_stack(cols)
def opfit(i,kind,fun=None):
 def prof(w):
  F=models(u[i],w,kind); return F,fit(F,v[i]);
 def r(w):
  F,c=prof(w);return (F@c-v[i]).ravel()
 opt=least_squares(r,[2.8 if kind.startswith('sin') else 1.4],bounds=(.4,4.5)); w=opt.x; return lambda X:models(X,w,kind)@prof(w)[1],w,np.sqrt(np.mean(opt.fun**2)),prof(w)[1]
rng=np.random.default_rng(710); ids=[np.arange(72),*[rng.permutation(72) for _ in range(8)],*[np.argsort(abs(z[:,j])) for j in range(4)]]
def cv(B,F,K=None,l=0):
 out=[]
 for k,ix in enumerate(ids):
  n=56 if 0<k<9 else 52; i,j=ix[:n],ix[n:]; sc=np.maximum(F[i].std(0),.05); f=F[i]/sc; g=F[j]/sc
  if K is None:
   c=fit(f,Y[i]) if l==0 else np.linalg.solve(f.T@f+l*np.eye(f.shape[1]),f.T@Y[i]); P=g@c
  else:
   ki=K[np.ix_(i,i)]+l*np.eye(len(i)); fi=solve(ki,f,assume_a='pos'); yi=solve(ki,Y[i],assume_a='pos'); c=fit(f.T@fi,f.T@yi); P=g@c+K[np.ix_(j,i)]@(yi-fi@c)
  e=np.abs(P-Y[j]); out.append([e[:,0].mean(),e[:,1:].mean()]);
 out=np.asarray(out); return np.round(np.r_[out[0],out[1:9].max(0),out[9:].ravel()],5)
ops={}
for family in ('sin','tanh','erf'):
 for sfx in ('','diag2','even','cross','quad','rad'):
  name=family+sfx; h,w,err,c=opfit(np.arange(len(u)),name); rs=[]
  for j in folds:
   hh,*_=opfit(np.setdiff1d(np.arange(len(u)),j),name);rs.extend((hh(u[j])-v[j]).ravel())
  print(name,w.round(4),round(err,6),round(np.sqrt(np.mean(np.square(rs))),6),cv(h(z),poly(h(z),2))); ops[name]=h
  if not sfx: print('affine coef',np.round(c[:5],3))
for name in ('sin','tanh','erf'):
 b=ops[name](z); p2=poly(b,2); c=fit(p2,Y); print('readout',name,'y beta',np.round(c[:,0],3),'edge',cv(b,p2));
 for k in (0,.18,.6):
  for x in (z,b):
   K=None if k==0 else np.exp(-k*cdist(x,x,'sqeuclidean')); print(name,'raw' if x is z else 'op',k,cv(b,p2,K,.07 if k else .025))
 for lab,F in [('bypass',np.c_[p2,z]),('three',np.c_[p2,np.column_stack([np.prod(b[:,j],1) for j in combinations(range(4),3)])]),('cubes',np.c_[p2,b**3]),('times z',np.c_[p2,z*b]),('rational',np.c_[p2,b/(1+b**2)]),('bounded quadratic',np.c_[lin(b),poly(np.tanh(b),2)[:,5:]])]:print(name,lab,cv(b,F,l=.012))
