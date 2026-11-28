import pickle,itertools,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True,linewidth=130)
def aff(x):return np.c_[np.ones(len(x)),x]
def coeff(x,y):return lstsq(x,y)[0]
def poly(x,d,sep=False):return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),k) if not sep or len(set(j))<=1])
p=s['panel_c'];u=p['u'];V1,V2=p['v1'],p['v2'];UV=np.r_[u,s['train']['x']]
for v in (V1,V2):
 print('v panel')
 fs={'aff':aff(u),'power2':poly(u,2),'diag2':poly(u,2,1),'diag3':poly(u,3,1),'diag4':poly(u,4,1),'diag5':poly(u,5,1),'aff+tan':np.c_[aff(u),np.tanh(u)],'aff+v2':np.c_[aff(u),V2], 'diag2+tan':np.c_[poly(u,2,1),np.tanh(u)],'diag2+abs':np.c_[poly(u,2,1),abs(u)]}
 for k,F in fs.items():
  if v is V2 and k=='aff+v2':continue
  c=coeff(F,v);R=v-F@c;h=np.sum(F*np.linalg.pinv(F).T,axis=1);print(k, np.mean(abs(R)),np.mean(abs(R)/np.maximum(1-h[:,None],1e-8)))
 # Shared scalar transforms, with a freely fit affine map before each output nonlinearity.
 for name,fn in [('sin',np.sin),('tanh',np.tanh),('sinh',np.sinh),('asinh',np.arcsinh),('atan',np.arctan),('softsign',lambda x:x/(1+abs(x)))]:
  train=[];errors=[]
  for k in range(4):
   A=aff(u);yy=v[:,k]
   def pred(q,A):return q[0]*fn(A@q[2:])+q[1]
   def dofit(I):
    ini=np.r_[1.8,0,coeff(A[I],yy[I])/1.8];return least_squares(lambda q:pred(q,A[I])-yy[I],ini,max_nfev=150).x
   I=np.arange(len(u));q=dofit(I);train.append(np.mean(abs(pred(q,A)-yy)));q=dofit(I[:18]);errors.append(np.mean(abs(pred(q,A[18:])-yy[18:])))
  print(name,np.round(train,4),np.round(errors,4))
 print('affine residual sv',np.linalg.svd(v-aff(u)@coeff(aff(u),v),compute_uv=False))
 print('affine+diag sq coeff', coeff(poly(u,2,1),v))
# Compare outer held-out errors across panel operators using regularized small-data smooth kernels.
a=s['panel_a'];b=s['panel_b'];a0=[];a1=[]
for i in range(4):
 uu=a['u'][:,i]; vv=a['v'][:,i];j,k=np.triu_indices(len(uu),1);sl=np.median((vv[j]-vv[k])/(uu[j]-uu[k]));bb=np.median(vv-sl*uu);ok=abs(vv-sl*uu-bb)<.01;bb,sl=coeff(aff(uu[ok,None]),vv[ok]);a0.append(bb);a1.append(sl)
a0=np.array(a0);a1=np.array(a1);B=coeff(aff(b['u']),b['v']);C1=coeff(aff(u),V1);C2=coeff(poly(u,3,1),V2)
x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];t=s['eval_times'];F=-np.expm1(-t[:,None]*np.array([.4121,1.8165]));Q=coeff(F,T.T).T;Y=np.c_[y,Q];Z={}
for k,z in [('raw',x),('a',x*a1+a0),('ai',(x-a0)/a1)]:
 Z[k]=z;Z[k+'b']=aff(z)@B
for k,z in list(Z.items()):
 Z[k+'c1']=aff(z)@C1;Z[k+'c2']=poly(z,3,1)@C2;Z[k+'+c2']=np.c_[z,poly(z,3,1)@C2]
folds=[np.arange(56,72),*np.array_split(np.random.default_rng(34).permutation(72),4)]
def fitcv(K):
 split=[]
 for J in folds:
  I=np.setdiff1d(np.arange(len(x)),J);split.append((I,J))
 allerrs=[]
 for reg in (1e-6,1e-4,.001,.01,.1):
  err=[]
  for I,J in split:
   P=K[np.ix_(J,I)]@solve(K[np.ix_(I,I)]+reg*np.eye(len(I)),Y[I],assume_a='pos');err.append(np.c_[abs(P[:,0]-y[J]),abs(P[:,1:]@F.T-T[J])])
  e=np.array([i.mean(0)[[0,-1]] for i in err]);allerrs.append(np.r_[e.mean(0),e.max(0)])
 return np.asarray(allerrs)
results=[]
for k,z in Z.items():
 z=z/np.maximum(z.std(0),1e-6);D=squareform(pdist(z,'sqeuclidean'))/z.shape[1]; G=(z@z.T)/z.shape[1]; R=np.sqrt(D)
 for typ,K in [(f'rbf{h}',np.exp(-D*h)) for h in (.15,.5,1.5,4)]+[(f'mat{h}',(1+R*h+(R*h)**2/3)*np.exp(-R*h)) for h in (1,2,4)]+[(f'poly{d}',(1+G)**d) for d in (2,3,4)]+[('add',1+sum(np.exp(-.5*(q[:,None]-q)**2) for q in z.T)+.25*G**2)]:
  E=fitcv(K)
  for j in range(5):results.append((E[j],k,typ,j))
for col in [0,1]:
 print('metric',col)
 best={}
 for e,k,typ,j in sorted(results,key=lambda r:r[0][col]):
  if k not in best:best[k]=(e,typ,j)
 for k,(e,typ,j) in list(best.items())[:12]:print(k,typ,j,e)
np.savez('/tmp/b02_coords.npz',x=x,y=y,T=T,t=t,F=F,Q=Q,**Z)
