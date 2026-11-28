import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; T=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; one=lambda u:np.c_[np.ones(len(u)),u]
pa=s['panel_a']; h=one(pa['u']); a=np.linalg.lstsq(h,pa['v'],rcond=None)[0]; keep=np.linalg.norm(h@a-pa['v'],axis=1)<3; a=np.linalg.lstsq(h[keep],pa['v'][keep],rcond=None)[0]; print('A',a,'res',np.linalg.norm(h[keep]@a-pa['v'][keep],axis=1).max()); z=(x-a[0])@np.linalg.inv(a[1:]); zd=(x-a[0])/a[1:].diagonal()
pb=s['panel_b']; b=np.linalg.lstsq(one(pb['u']),pb['v'],rcond=None)[0]; print('B singulars',np.linalg.svd(b[1:],compute_uv=False))
def q(p):return -np.expm1(-t[:,None]*p)
def amp(p):return np.linalg.lstsq(q(p),T[:52].T,rcond=None)[0].T
def err(p):return (amp(p)@q(p).T-T[:52]).ravel()
p=least_squares(err,[.35,1.8],bounds=(.01,10),ftol=1e-12).x; AA=np.linalg.lstsq(q(p),T.T,rcond=None)[0].T; Y=np.c_[y,AA]; print('rates',p,'res',np.std(err(p)),'amps std',AA.std(0),'means',AA.mean(0)); print('Y linear corr',np.corrcoef(Y.T))
frames={'raw':x,'cal':z,'diag':zd,'Braw':one(x)@b,'Bcal':one(z)@b,'Bdiag':one(zd)@b,'calBinv':(z-b[0])@np.linalg.inv(b[1:])}
for name,U in frames.items():
 print(name,'range',np.min(U,0).round(2),np.max(U,0).round(2),'corr',np.corrcoef(np.c_[U,Y].T)[:4,4:].round(2).T)
 for d in [1,2,3,4]:
  u=PolynomialFeatures(d).fit_transform(U); sd=np.sqrt(np.mean(u[:52]**2,0)); u/=np.maximum(sd,1e-7)
  out=[]
  for r in [1e-6,.001,.03,.3,2]:
   pred=Ridge(alpha=r,fit_intercept=False).fit(u[:52],Y[:52]).predict(u[52:]); out.append(np.r_[np.mean(abs(pred-Y[52:]),0),np.mean(abs(pred[:,1:]@q(p).T-T[52:]))])
  print(d, np.array(out).min(0).round(4))
