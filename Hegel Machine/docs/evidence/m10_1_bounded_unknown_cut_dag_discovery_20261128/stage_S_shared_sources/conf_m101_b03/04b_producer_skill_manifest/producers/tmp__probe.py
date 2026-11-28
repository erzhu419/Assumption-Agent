import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.kernel_ridge import KernelRidge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=tr['x'],tr['y'],tr['traj']; t=s['eval_times']; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=145)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(f,z):return np.linalg.lstsq(f,z,rcond=None)[0]
a=s['panel_a']; u=aff(a['u']); v=a['v']; rng=np.random.default_rng(3); best=None
for _ in range(500):
 i=rng.choice(len(u),6,replace=False); b=ls(u[i],v[i]); e=np.linalg.norm(u@b-v,axis=1); score=np.median(e)
 if best is None or score<best[0]:best=(score,b)
e=np.linalg.norm(u@best[1]-v,axis=1); keep=e<4*np.median(e); A=ls(u[keep],v[keep]); p=s['panel_b']; B=ls(aff(p['u']),p['v']); print('A inliers',sum(keep),'resids',np.sort(np.linalg.norm(u@A-v,axis=1)),'A',A,'B',B); print('time',t,'telemetry',s['telemetry_vec'],'summary',tr['summary']); print('x,y,T',np.c_[x,y,T][::4]);
def modes(k):return np.c_[np.ones(len(t)),np.exp(-k*t)]
def mse(k):
 f=modes(k); return np.mean((f@ls(f,T[:n].T)-T[:n].T)**2)
k=minimize_scalar(mse,bounds=(.01,20),method='bounded').x; f=modes(k); z=ls(f,T.T).T; print('decay',k,'train and val reconstruction',np.mean(abs(z[:n]@f.T-T[:n])),np.mean(abs(z[n:]@f.T-T[n:])),'modes quantiles',np.quantile(z,[0,.25,.5,.75,1],axis=0)); Z=np.c_[y,z];
base={'x':x,'a':aff(x)@A,'ai':(x-A[0])@np.linalg.inv(A[1:]),'b':aff(x)@B,'bi':(x-B[0])@np.linalg.inv(B[1:])}; base.update({k+'b':aff(base[k])@B for k in ['a','ai']});base.update({k+'bi':(base[k]-B[0])@np.linalg.inv(B[1:]) for k in ['a','ai']});
def err(p):return np.mean(abs(p-Z[n:]),axis=0)
for name,q in base.items():
 print('features',name,'ranges',q.min(0),q.max(0));
 for d in (1,2,3):
  g=PolynomialFeatures(d).fit_transform(q); print(' poly',d,err(g[n:]@ls(g[:n],Z[:n])))
 for transform in (np.tanh,np.sin,np.abs):
  h=np.c_[aff(q),transform(q)]; print(' link',transform.__name__,err(h[n:]@ls(h[:n],Z[:n])))
for gamma in [.025,.075,.2,.5,1]:
 for alpha in [.00001,.001,.01,.1]:
  g=StandardScaler().fit(x[:n]).transform(x); model=KernelRidge(alpha=alpha,kernel='rbf',gamma=gamma).fit(g[:n],Z[:n]); print('rbf',gamma,alpha,err(model.predict(g[n:])))
