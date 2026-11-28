import numpy as np,pickle,sys
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_m102_c01_batch_01_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a,b,c=[s['panel_'+k] for k in 'abc']; x,y,T=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; cal=old._calibration(a); print('calibration',cal); print('temporal SVD',np.linalg.svd(T,compute_uv=False)); print('temporal PCA',np.linalg.svd(T,full_matrices=False)[2][:2]);
def fit(F,Y): return np.linalg.lstsq(F,Y,rcond=None)[0]
def ext(z): return np.c_[np.ones(len(z)),z]
def quad(z):
 i,j=np.triu_indices(z.shape[1]);return z[:,i]*z[:,j]
def test(f,Y,N=52):
 w=fit(f[:N],Y[:N]); err=f@w-Y
 return np.mean(abs(err[:N])),np.mean(abs(err[N:]))
ops={}
for label,p,k in [('b',b,'v'),('c1',c,'v1'),('c2',c,'v2')]:
 u,v=p['u'],p[k]; best=[]
 fs={'linear':lambda z:z,'quadratic':lambda z:np.c_[z,quad(z)],'tanh':lambda z,k:np.tanh(k*z),'sin':lambda z,k:np.sin(k*z),'rational':lambda z,k:np.c_[z,z*z/(1+k*z*z)],'tanhlin':lambda z,k:np.c_[z,np.tanh(k*z)],'linT':lambda z,k:np.arctanh(np.clip(k*z,-.9999,.9999))/k}
 for n,f in fs.items():
  def loo(F,Y):
   Q=np.linalg.qr(F)[0];return np.sqrt(np.mean(((Y-Q@(Q.T@Y))/(1-(Q*Q).sum(1))[:,None])**2))
  if n in ('linear','quadratic'): q=f; K=0
  elif n=='linT':
   K=minimize_scalar(lambda k:loo(ext(u),f(v,k)),bounds=(.001,.999/abs(v).max()),method='bounded').x;q=lambda z:z
  else:
   K=minimize_scalar(lambda k:loo(ext(f(u,k)),v),bounds=(.02,2.8),method='bounded').x;q=lambda z,k=K,f=f:f(z,k)
  F=ext(q(u)); score=loo(F,v); w=fit(F,v); best.append((score,n,K)); ops[label+n]=lambda z,q=q,w=w:ext(q(z))@w
 print(label,sorted(best))
coords={'x':x,'uncal':(x-cal[1])/cal[0],'forward':x*cal[0]+cal[1]}
for n,z in list(coords.items()):
 for m in ('blinear','btanh','c1linear','c1tanh','c2rational'):
  coords[n+'_'+m]=ops[m](z)
pca=np.linalg.svd(T,full_matrices=False)[2]; A=T@pca[:2].T
res=[]
for n,z in coords.items():
 for k,F in dict(lin=z,tanh=np.tanh(z),quad=np.c_[z,quad(z)],tanhquad=np.c_[np.tanh(z),quad(z)]).items():
  res.append((n,k,test(ext(F),y),test(ext(F),T),test(ext(np.c_[F,A]),y)))
for j in [2,3,4]:
 print('ranking',j);print(*sorted(res,key=lambda e:e[j][1])[:10],sep='\n')
