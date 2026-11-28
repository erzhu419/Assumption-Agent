exec(open('/tmp/graph_probe.py').read().split("for key in ('v1','v2'):")[0])
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); raw=raw[:n]; x=x[:n]; idx=np.arange(n); rng=np.random.default_rng(18); folds=[idx[52:]]+list(np.array_split(rng.permutation(n),4)); folds += [np.argsort(np.linalg.norm(raw,axis=1))[-14:]]
p=s['panel_a']; A=np.c_[np.ones(len(p['u'])),p['u']]; keep=np.ones(len(A),bool)
for _ in range(6):
 ca=np.linalg.lstsq(A[keep],p['v'][keep],rcond=None)[0]; r=np.linalg.norm(A@ca-p['v'],axis=1); keep=r<max(.001,4*np.median(r))
inv=np.linalg.solve(ca[1:].T,(raw-ca[0]).T).T
W={'raw':raw,'cal':x,'invA':inv}; W.update({nm+suffix:w@M+c for nm,w in list(W.items()) for suffix,M,c in [('B',b[1:],b[0]),('iB',np.linalg.inv(b[1:]),-b[0]@np.linalg.inv(b[1:]))]})
print('y-traj correlation',np.corrcoef(Y.T)[0].round(3),'traj svd',np.linalg.svd(Y[:,1:],compute_uv=False).round(4)); print('panel B offset',b[0].round(4))
def norm(K):
 K=K-K.mean(0)-K.mean(1)[:,None]+K.mean(); return K/max(np.trace(K)/len(K),1e-15)
def Kf(F):return norm(F@F.T)
poly=norm((1+raw@raw.T)**2); pool={}
for nm,w in W.items():
 for c in (.65,1.3,1.7,2.5,3.5):
  for phi,fn in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('rat',lambda z:z/np.sqrt(1+z*z))]:
   z=fn(c*w); d=w@w.T; g=z@z.T
   pool[nm,phi,c,'add']=Kf(z)
   pool[nm,phi,c,'quad']=norm((1+g)**2)
   pool[nm,phi,c,'cross']=norm((1+d)*(1+g))
   pool[nm,phi,c,'separable']=norm(sum((1+np.outer(z[:,i],z[:,i]))**2 for i in range(4)))
  d=cdist(w,w,'sqeuclidean'); g=np.exp(-d*c/4); pool[nm,'rbf',c,'joint']=norm(g)
  g=sum(np.exp(-c*cdist(w[:,i:i+1],w[:,i:i+1],'sqeuclidean')) for i in range(4)); pool[nm,'rbf',c,'add']=norm(g); pool[nm,'rbf',c,'pairs']=norm(g*g)
 for d in (3,4,5,6):pool[nm,'poly',d,'joint']=norm((1+w@w.T/2)**d)
for name,base in list(pool.items()):
 if name[0]=='cal':
  matching=[pool[(nm,*name[1:])] for nm in W]
  pool['avg',*name[1:]]=np.mean(matching,axis=0)
L=np.array([1e-6,1e-5,1e-4,.001,.01,.05,.2]); tridx=[np.setdiff1d(idx,te) for te in folds]
def evalk(K):
 scores=[]
 for tr,te in zip(tridx,folds):
  G=K[np.ix_(tr,tr)]; J=K[np.ix_(te,tr)]; mu=G.mean(0); J=J-mu-J.mean(1)[:,None]+mu.mean(); G=G-mu-mu[:,None]+mu.mean(); e,V=eigh(G); Z=V.T@(Y[tr]-Y[tr].mean(0)); J=J@V; pr=np.einsum('ij,ljk->lik',J,Z[None,:,:]/(np.maximum(e,0)[None,:,None]+L[:,None,None]))+Y[tr].mean(0); err=np.abs(pr-Y[te]);scores.append(np.c_[err[:,:,0].mean(1),err[:,:,1:].mean((1,2))])
 return np.stack(scores,axis=1)
results=[]
for name,K in pool.items():
 for mix in (0,.2,.7,2,8):
  sc=evalk(K+mix*poly)
  for lam,v in zip(L,sc):results.append((v[:5,0].max(),v[0,0],v[:5,0].mean(),v[5,0],v[:,1].max(),name,mix,lam,v[:,0].round(4).tolist()))
for label,f in [('worst',lambda r:r[0]),('official',lambda r:r[1]),('average',lambda r:r[2]),('extrap',lambda r:max(r[0],r[3]))]:
 print(label)
 for row in sorted(results,key=f)[:7]:print(row)
