import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; t=s['eval_times']; q=tr['traj']; n=52
np.set_printoptions(precision=4,suppress=True)
def design(x): return np.c_[np.ones(len(x)),x]
def affine(p,robust=False):
 u=design(p['u']); v=p['v']; ix=np.arange(len(u));
 for j in range(8 if robust else 1):
  c=np.linalg.lstsq(u[ix],v[ix],rcond=None)[0]; r=np.linalg.norm(u@c-v,axis=1); ix=np.argsort(r)[:39]
 if robust:
  print('A residual',np.sort(r)); ix=np.where(r<max(.003,4*np.median(r)))[0]; c=np.linalg.lstsq(u[ix],v[ix],rcond=None)[0]
 return c
A=affine(s['panel_a'],True); B=affine(s['panel_b']); print('A,B',A,B,sep='\n'); print('times',t,'telemetry',s['telemetry_vec'])
def basis(r): return -np.expm1(-t[:,None]*r)
def coeff(h): return np.linalg.lstsq(h,q[:n].T,rcond=None)[0]
r=np.sort(least_squares(lambda r:(basis(r)@coeff(basis(r))-q[:n].T).ravel(),[.5,2.],bounds=(.02,20)).x); h=basis(r); m=q@np.linalg.pinv(h.T); print('rates',r,'recon',np.mean(abs(m@h.T-q),0),'mode stats',m.mean(0),m.std(0)); Y=np.c_[tr['y'],m]
def fw(z,M): return design(z)@M
def inv(z,M): return (z-M[0])@np.linalg.inv(M[1:])
xs={'X':x,'A':fw(x,A),'a':inv(x,A)}
for k,z in list(xs.items()): xs[k+'B']=fw(z,B); xs[k+'b']=inv(z,B)
for k,z in xs.items():
 print('\n',k,'lin slopes',np.linalg.lstsq(design(z[:n]),Y[:n],rcond=None)[0].T)
 feats=[('lin',design(z)),('quad',PolynomialFeatures(2).fit_transform(z)),('cubic',PolynomialFeatures(3).fit_transform(z)),('add',np.c_[design(z),z**2,z**3,np.sin(z),np.cos(z),np.tanh(z)]),('addquad',np.c_[PolynomialFeatures(2).fit_transform(z),z**3,np.sin(z),np.cos(z),np.tanh(z)])]
 for label,F in feats:
  scale=np.maximum(F[:n].std(0),.01); scale[0]=1; F=F/scale; R=F[:n].T@F[:n]; P=F[n:]@np.linalg.solve(R+np.eye(len(R))*1e-3,F[:n].T@Y[:n]); print(label,np.mean(abs(P-Y[n:]),0),np.mean(abs(P[:,1:]@h.T-q[n:])).round(5),end='; ')
 print()
