import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=145)
def poly(x,d):return PolynomialFeatures(d).fit_transform(x)
for key in ['panel_a','panel_b','panel_c']:
 p=s[key];u=p['u']
 for k,v in p.items():
  if k=='u':continue
  print(key,k)
  for d in range(1,4 if key=='panel_a' else 3):
   X=poly(u,d); m=np.linalg.norm(v,axis=1)<3 if key=='panel_a' else np.ones(len(v),bool)
   X=X[m];V=v[m]; B=lstsq(X,V)[0]; e=X@B-V; h=np.sum(X*np.linalg.pinv(X).T,axis=1); loo=e/np.maximum(1-h[:,None],1e-9)
   print('degree',d,'fit & LOO',np.mean(abs(e)),np.mean(abs(loo)))
   if d==2:print(B)
x=s['train']['x'];t=s['train']['traj'];y=s['train']['y']; V=np.c_[y,t]
for d in range(1,6):
 X=poly(x,d)
 for l in [0,.001,.03,.3]:
  A=X[:52];B=lstsq(np.r_[A,np.sqrt(l)*np.eye(X.shape[1])],np.r_[V[:52],np.zeros((X.shape[1],V.shape[1]))])[0]; p=X@B
  print('forward poly',d,l,np.mean(abs(p[:52]-V[:52]),axis=0)[[0,1,8]],np.mean(abs(p[52:]-V[52:]),axis=0)[[0,1,8]])
from scipy.interpolate import RBFInterpolator
for k in ['thin_plate_spline','cubic','quintic','gaussian','inverse_multiquadric']:
 for e in [.3,.6,1]:
  for sm in [0,.001,.01]:
   p=RBFInterpolator(x[:52],V[:52],kernel=k,epsilon=e,smoothing=sm)(x[52:]); print('rbf',k,e,sm, np.mean(abs(p[:,0]-y[52:])),np.mean(abs(p[:,1:]-t[52:])))
