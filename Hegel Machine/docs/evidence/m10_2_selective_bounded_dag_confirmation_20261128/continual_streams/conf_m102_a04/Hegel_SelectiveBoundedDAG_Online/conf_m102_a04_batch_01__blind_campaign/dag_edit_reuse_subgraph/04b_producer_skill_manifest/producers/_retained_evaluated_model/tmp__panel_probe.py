import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def fit(F,y):
 b=np.linalg.lstsq(F,y,rcond=1e-10)[0];r=y-F@b;return b,np.sqrt(np.mean(r*r,axis=0))
for k in ['panel_a','panel_b','panel_c']:
 p=s[k];u=p['u'];F=np.c_[np.ones(len(u)),u];u2=u*u
 for t in [k for k in p if k!='u']:
  y=p[t]; print(k,t,'linear',fit(F,y));
  if t!='v1' and k!='panel_b':continue
  pairs=list(itertools.combinations(range(4),2));cross=np.column_stack([u[:,i]*u[:,j] for i,j in pairs]); print('diag quadratic',fit(np.c_[F,u2],y)); print('full quadratic',fit(np.c_[F,u2,cross],y)); print('separate cubic',fit(np.c_[F,u2,u**3],y)); print('rational diagonal',fit(np.c_[F,u2,u2/(1+.27*u2)],y)); print('targets svd',np.linalg.svd(y-y.mean(0),compute_uv=False));
  for fam in [lambda x:x*x, lambda x:np.abs(x),lambda x:x**3,lambda x: np.tanh(1.25*x),lambda x:x/(1+1.25*abs(x)),lambda x: x/(1+1.25*x*x),lambda x:np.sin(1.25*x),lambda x: np.cos(1.25*x)]:
   f=fam(u);print('family',fit(np.c_[F,f],y)[1]);
  print('decomposition left singular modes after affine',np.linalg.svd(y-F@fit(F,y)[0],compute_uv=False))
