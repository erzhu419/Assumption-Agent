import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.linalg import svd
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=120)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; Y=np.c_[y,z]; err=lambda a,b:np.mean(abs(a-b),axis=0)
for name in ['panel_a','panel_b','panel_c']:
 p=s[name]; u=p['u'];
 for vn,v in p.items():
  if vn=='u':continue
  keep=np.linalg.norm(v,axis=1)<3 if name=='panel_a' else np.ones(len(u),bool); u0=u[keep]; v=v[keep]; print(name,vn)
  for d in (1,2,3):
   U=PolynomialFeatures(d).fit_transform(u0); A=np.linalg.lstsq(U,v,rcond=None)[0]; resid=U@A-v; print(d, np.sqrt(np.mean(resid**2,axis=0)), 'loo',np.sqrt(np.mean((resid/(1-np.diag(U@np.linalg.pinv(U)))[:,None])**2,axis=0)) if len(u0)>U.shape[1] else None); print(A if d==1 else '')
print('traj singular',np.linalg.svd(z,compute_uv=False));print('y correlations',np.corrcoef(np.c_[x,Y].T)[4]); print('traj right singulars',svd(z)[2][:4]);
for d in range(1,5):
 U=PolynomialFeatures(d).fit_transform(x)
 for a in (.001,.1,1.):
  P=Ridge(alpha=a).fit(U[:52],Y[:52]).predict(U[52:]); print('poly',d,a,err(P,Y[52:]))
D=cdist(x,x,'sqeuclidean');
for l in (.5,1,2,4,8):
 K=np.exp(-D/(2*l*l))
 for a in (1e-5,.001,.01):
  P=K[52:,:52]@np.linalg.solve(K[:52,:52]+a*np.eye(52),Y[:52]); print('rbf',l,a,err(P,Y[52:])[[0,-1]])
