import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(a,d): return PolynomialFeatures(d).fit_transform(a)
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']
 for name,v in p.items():
  if name=='u': continue
  z=poly(u,1); mask=np.ones(len(u),bool)
  if k=='panel_a':
   for j in range(4):
    e=np.linalg.norm(v-z@fit(z[mask],v[mask]),axis=1); mask=np.argsort(e)[:36]
   e=np.linalg.norm(v-z@fit(z[mask],v[mask]),axis=1); print('sorted residual',np.sort(e)); mask=e<0.15
  print(k,name,'n',np.count_nonzero(mask))
  coef=fit(z[mask],v[mask]);print(coef)
  for d in (1,2,3):
   z=poly(u[mask],d); vv=v[mask]; h=np.sum(z*np.linalg.pinv(z).T,axis=1); e=vv-z@fit(z,vv); print(d,'in rms',np.sqrt(np.mean(e**2,axis=0)),'loo rms',np.sqrt(np.mean((e/np.maximum(1-h,1e-8)[:,None])**2,axis=0)))
print('summary',tr['summary'],Y[:,0].mean(),Y[:,0].std(),Y[:,-1].mean(),Y[:,-1].std())
print('trajectory singular values',np.linalg.svd(Y[:,1:],compute_uv=False))
idx=np.arange(52); va=np.arange(52,72)
def score(y):
 e=np.abs(Y[va]-y); return np.r_[e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]
for d in (1,2,3):
 z=poly(x,d); print('poly',d,score(z[va]@fit(z[idx],Y[idx])))
D=cdist(x,x,'sqeuclidean')
for g in (.03,.1,.25,.5,1,2):
 K=np.exp(-D*g); print('rbf',g,[(a,score(K[np.ix_(va,idx)]@solve(K[np.ix_(idx,idx)]+a*np.eye(52),Y[idx],assume_a='pos'))) for a in (1e-5,.001,.01)])
print('y normalized linear coefficients',fit(poly(x,1),Y)); print('label-traject correlation',np.corrcoef(Y.T)[0]); print('y from trajectories',score(np.c_[poly(Y[va,1:],1)@fit(poly(Y[idx,1:],1),Y[idx,0]),Y[va,1:]]))
