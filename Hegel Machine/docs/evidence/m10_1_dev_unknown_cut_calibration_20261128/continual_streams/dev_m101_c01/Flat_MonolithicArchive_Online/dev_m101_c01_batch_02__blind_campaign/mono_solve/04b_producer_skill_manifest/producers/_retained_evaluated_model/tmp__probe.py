import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
np.set_printoptions(precision=5,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; d=s['train']; z=d['traj']; y=d['y']
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']
 for j,v in p.items():
  if j=='u':continue
  print(k,j)
  good=np.ones(len(u),bool)
  for a in range(3 if k=='panel_a' else 1):
   X=np.c_[np.ones(len(u)),u]; b=np.linalg.lstsq(X[good],v[good],rcond=None)[0]; e=np.linalg.norm(v-X@b,axis=1)
   if k=='panel_a':good=e<2
  print('retained',sum(good),'linear',b,'affine residuals',np.round(np.sort(e),4))
  for name,F in [('diag2',np.c_[X,u**2]),('diag3',np.c_[X,u**2,u**3]),('poly2',PolynomialFeatures(2).fit_transform(u)),('sin',np.c_[X,np.sin(1.7*u),np.cos(1.7*u)]),('cube',np.c_[X,u**3])]:
   q=np.linalg.lstsq(F[good],v[good],rcond=None)[0]; h=np.sum(np.linalg.svd(F[good],full_matrices=False)[0]**2,axis=1); print(name,np.sqrt(np.mean(((v[good]-F[good]@q)/(1-h[:,None]))**2)))
U,S,V=np.linalg.svd(z,full_matrices=False); print('uncentered singular',S,'basis', V[:2]);
exp=lambda r:-np.expm1(-r*t)
sin=lambda w:np.sin(w*t)
fun={'es':(lambda a:np.array([exp(a[0]),sin(a[1])]).T,[.5,1.1]),'eds':(lambda a:np.array([exp(a[0]),sin(a[1])*np.exp(-a[2]*t)]).T,[.5,1.1,.1]),'ss':(lambda a:np.array([sin(a[0]),sin(a[1])]).T,[.4,1.1]),'sc':(lambda a:np.array([sin(a[0]),1-np.cos(a[0]*t)]).T,[1.1]),'ep':(lambda a:np.array([exp(a[0]),t]).T,[.5]),'escos':(lambda a:np.array([exp(a[0]),sin(a[1]+a[2]*t)-np.sin(a[1])]).T,[.5,0,1.1]),'dsds':(lambda a:np.array([sin(a[0])*np.exp(-a[2]*t),sin(a[1])*np.exp(-a[3]*t)]).T,[.4,1.1,.1,.1])}
for k,(f,p) in fun.items():
 def r(a):
  B=f(a); return (B@np.linalg.lstsq(B,z.T,rcond=None)[0]-z.T).ravel()
 a=least_squares(r,p,max_nfev=1000).x; print(k,a,np.sqrt(np.mean(r(a)**2)))
