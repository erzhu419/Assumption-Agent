from ops import *
from scipy.spatial.distance import cdist
T=Y[:,1:]; t=s['eval_times']; V=np.linalg.svd(T[:52],full_matrices=False)[2][:2]; Z=np.c_[Y[:,0],T@V.T]
dicts={}
for idx,(p,key,n) in enumerate([('panel_b','v','erf'),('panel_c','v1','asinh'),('panel_c','v2','log2')]):
 u,v=s[p]['u'],s[p][key]; _,k,b=fit(u,v,n,True); dicts['bcd'[idx]]=lambda u,n=n,k=k,b=b:design(u,n,k,True)@b
bases={'x':x}
for k,f in dicts.items():bases[k]=f(x)
for k,f in dicts.items():
 for l,g in dicts.items():
  if k!=l:bases[k+l]=g(f(x))
for k in ['bc','bd','cb','cd','db','dc']:
 l=next(iter(set('bcd')-set(k)));bases[k+l]=dicts[l](bases[k])
for k in list(bases):
 if k!='x':bases['x+'+k]=np.c_[x,bases[k]]
for k in ['b+c','b+d','c+d','x+b+c+d']:
 bases[k]=np.column_stack([bases[l] for l in k.split('+')])

def evaluate(K):
 K=K/K.diagonal().mean();A=K[:52,:52];B=K[52:,:52]; E,Q=np.linalg.eigh(A);P=(B@Q)[:,:,None]*(Q.T@Z[:52])[None,:,:];
 out=[]
 for reg in [1e-9,1e-7,1e-5,1e-3,.01,.1]:
  R=(P/np.maximum(E+reg,reg)[None,:,None]).sum(1)-Z[52:];out.append([abs(R[:,0]).mean(),abs(R[:,1:]@V).mean(),reg]);
 return out
for name,a in bases.items():
 a=(a-a[:52].mean(0))/a[:52].std(0); results=[]
 for trans,g in [('raw',lambda a:a),('asinh',lambda a:np.arcsinh(2*a)),('tanh',np.tanh),('combo',lambda a:np.c_[a,np.arcsinh(2*a),np.log1p(a*a)])]:
  b=g(a); G=b@b.T/b.shape[1]; D=cdist(b,b,'sqeuclidean')/b.shape[1];
  for typ,K in [(str(i),(1+G)**i) for i in [1,2,3,4,5]]+[(f'rbf{r}',np.exp(-r*D)) for r in [.03,.1,.3,1]]:
   results += [(*v,trans,typ) for v in evaluate(K)]
 print(name,'y',min(results,key=lambda v:v[0]),'t',min(results,key=lambda v:v[1]),'both',min(results,key=lambda v:max(v[:2])))
