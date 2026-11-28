from graph import *
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from itertools import combinations
from sklearn.linear_model import lars_path
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=13).split(X)]
for n,d in routes(X):
 F=PolynomialFeatures(2).fit_transform(d); a=ls(F[:52],Y[:52]);print('P2',n,np.round(a,3))
print('A',ab,'B',B)
V=np.linalg.svd(T[:52],full_matrices=False)[2][:2]; P=T@V.T
for k,F in [('P',P),('PX',np.c_[P,X]),('PC',np.c_[P,C1(X),C2(X)]),('PXA',np.c_[P,X,aug(X)@B])]:
 for deg in [1,2]:
  q=PolynomialFeatures(deg).fit_transform(F);print('mode',k,deg,np.mean(abs(q[52:]@ls(q[:52],Y[:52])-Y[52:])))
def features(d,z,w,kind):
 blocks=[]; labels=[]
 def add(n,f):
  if f.ndim==1:f=f[:,None]
  blocks.append(f);labels.extend([n+':'+str(j) for j in range(f.shape[1])])
 add('base',d);add('C1',z);add('C2',w)
 add('quadratic',PolynomialFeatures(2,include_bias=False).fit_transform(d)[:,4:]);add('cubic',PolynomialFeatures(3,include_bias=False).fit_transform(d)[:,14:]);
 for n,f in [('d',d),('C1',z),('C2',w),('sum',z+w),('diff',z-w),('dc1',d+z),('dc2',d+w),('xprod',z*w)]:
  if kind and n in ('d','C1','C2'):
   f=np.column_stack([f]+[f[:,i]+sg*f[:,j] for i,j in combinations(range(4),2) for sg in [-1,1]])
  for k in [.5,1,1.7,2.5]:
   for tag,v in [('sin',np.sin(k*f)),('cos',np.cos(k*f)-1),('tanh',np.tanh(k*f)),('soft',f/(1+k*abs(f))),('rat',f/(1+k*f*f)),('evenrat',f*f/(1+k*f*f) )]:add(n+tag+str(k),v)
  for power in [2,3]:add(n+'**'+str(power),f**power)
  add(n+'abs',abs(f))
  if n not in ('sum','diff','dc1','dc2','xprod'):add(n+'cross',np.column_stack([f[:,i]*f[:,j] for i,j in combinations(range(f.shape[1]),2)]))
 for n,a,b in [('z*w',z,w),('z*d',z,d),('w*d',w,d)]:add(n,(a[:,:,None]*b[:,None,:]).reshape(len(d),-1))
 return np.column_stack(blocks),labels
res=[]
for n,d in routes(X):
 z,w=C1(d),C2(d)
 for kind in [0,1]:
  F,labels=features(d,z,w,kind)
  results=[]
  for i,j in splits:
   mu=F[i].mean(0);sd=F[i].std(0)+1e-9;p=(F-mu)/sd; p=np.clip(p,-30,30); y=Y[i]-Y[i].mean()
   k=p@p.T/p.shape[1]; a,b=eigh(k[np.ix_(i,i)]);proj=b.T@y; cross=k[np.ix_(j,i)]@b
   row=[np.mean(abs(cross@(proj/(a+l))+Y[i].mean()-Y[j])) for l in [1e-6,1e-5,1e-4,.001,.01,.1]]
   al,ac,co=lars_path(p[i],y,method='lasso',max_iter=120); preds=p[j]@co+Y[i].mean(); ix=[5,10,18,30,45,65];row+= [np.mean(abs(preds[:,min(t,preds.shape[1]-1)]-Y[j])) for t in ix];results.append(row)
   if len(results)==1:
    for t in [10,30]:
     cf=co[:,min(t,co.shape[1]-1)]; print('sparse',n,kind,t,[(labels[h],round(cf[h],3)) for h in np.argsort(abs(cf))[-6:][::-1]])
  es=np.array(results).T
  for m,e in enumerate(es):res.append((max(e),n,kind,m,e))
print('grammar ranking')
for r in sorted(res,key=lambda r:r[0])[:18]: print(r)
