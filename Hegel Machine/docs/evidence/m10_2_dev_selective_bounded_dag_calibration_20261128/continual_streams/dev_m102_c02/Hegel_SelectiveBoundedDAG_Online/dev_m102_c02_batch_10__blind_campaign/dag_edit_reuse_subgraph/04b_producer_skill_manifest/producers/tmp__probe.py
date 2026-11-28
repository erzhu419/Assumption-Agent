import numpy as np,pickle,itertools
from scipy.linalg import svd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ix],axis=1) for e in range(1,d+1) for ix in itertools.combinations_with_replacement(range(4),e)])
def check(D,Y):
 b=np.linalg.lstsq(D,Y,rcond=None)[0]; R=Y-D@b; h=np.sum(D*np.linalg.pinv(D).T,axis=1); return np.round(np.mean(abs(R),axis=0),5),np.round(np.mean(abs(R/np.maximum(1e-12,1-h[:,None])),axis=0),5)
a=s['panel_a']; U,V=a['u'],a['v']; ii,jj=np.triu_indices(len(U),1); dx=U[jj]-U[ii]; a1=np.array([np.median(((V[jj]-V[ii])[:,k]/dx[:,k])[abs(dx[:,k])>.1*np.ptp(U[:,k])]) for k in range(4)]); a0=np.median(V-U*a1,axis=0); R=V-U*a1-a0; print('A maps',a0,a1,'resid quantiles',np.quantile(abs(R),[0,.25,.5,.75,.85,.9,.95,.99,1],axis=0)); print('A row medres',np.sort(np.median(abs(R),axis=1)))
for k in range(4):
 keep=abs(R[:,k])<6*np.median(abs(R[:,k])); a0[k],a1[k]=np.linalg.lstsq(aff(U[keep,k]),V[keep,k],rcond=None)[0]
X=s['train']['x']; Z=(X-a0)/a1; Y=np.c_[s['train']['y'],s['train']['traj']]; print('A final',a0,a1,'Z range',Z.min(0),Z.max(0),'singular values traj',svd(Y[:,1:],compute_uv=False))
def forms(u):
 yield 'aff',aff(u)
 for deg in [2,3,4,5,6,7]: yield 'sep'+str(deg),np.c_[aff(u),*[u**e for e in range(2,deg+1)]]
 for deg in [2,3]: yield 'poly'+str(deg),poly(u,deg)
 for e in [2,3,5]: yield 'pow'+str(e),np.c_[aff(u),u**e]
 for a in [.125,.25,.5,1,1.25,2,4]:
  yield 'rat'+str(a),aff(u)/(1+a*np.sum(u*u,axis=1,keepdims=True))
  yield 'seprat'+str(a),np.c_[aff(u),u/(1+a*u*u)]
  yield 'tanh'+str(a),np.c_[aff(u),np.tanh(a*u)]
  yield 'sin'+str(a),np.c_[aff(u),np.sin(a*u)]
for p,t in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][t]; rows=[]
 for n,d in forms(u):
  if d.shape[1]>=len(u)-2:continue
  f,l=check(d,v); rows.append((np.mean(l),n,f,l))
 print(p,t,sorted(rows,key=lambda x:x[0])[:9]); b=np.linalg.lstsq(aff(u),v,rcond=None)[0]; print('affine row resid',np.sort(np.linalg.norm(v-aff(u)@b,axis=1)))
print('LABELS')
for n,d in list(forms(Z))+[('x',aff(X))]:
 if d.shape[1]>=45:continue
 b=np.linalg.lstsq(d[:52],Y[:52],rcond=None)[0]; er=np.mean(abs(Y[52:]-d[52:]@b),axis=0); print(n,np.round([er[0],er[1:].mean(),er[-1]],4))
