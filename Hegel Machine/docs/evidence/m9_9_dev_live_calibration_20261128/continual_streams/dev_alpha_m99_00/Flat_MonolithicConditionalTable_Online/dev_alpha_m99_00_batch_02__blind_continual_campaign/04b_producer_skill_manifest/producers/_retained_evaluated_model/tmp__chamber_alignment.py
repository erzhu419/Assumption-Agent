import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_alpha_m99_00_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=a['dataset']; tr,te,d=ds['train'],ds['test'],a['campaign_diagnostics']; p=np.load('/tmp/chamber_probe.npz'); globals().update({k:p[k] for k in p.files}); n=len(tr['lab_id']); nd=len(lab); original=x.copy(); mechanical=y-E-g[:,None]*np.r_[1,t]; Y=y-mechanical
T=np.r_[1,t]; full=np.ones(nd,bool)
def design(rate):return np.c_[T,np.r_[rate,-np.expm1(-rate*t)]]
def fit(A,mask):
 A=A[mask]; yy=Y[mask]
 def solve(q):
  B=design(np.exp(q[0])); L=(A[:,None,:,None]*B[None,:,None,:]).reshape(yy.size,-1); coef=np.linalg.lstsq(L,yy.ravel(),rcond=None)[0]; return L@coef-yy.ravel(),coef.reshape(A.shape[1],2)
 q=least_squares(lambda q:solve(q)[0], [0.],bounds=(-3,3)).x; return np.exp(q[0]),solve(q)[1]
for name,x in [('prescribed-panel',original),('diagnostics-panel',m._features(d)),('all-panel',m._features(tr,te,d)[-nd:]),('train-plus-diagnostics',m._features(tr,d)[-nd:])]:
 g1=theta[2]+x@theta[3:]; print(name,'feature RMSE vs prescribed',np.sqrt(np.mean((x-original)**2)), 'missing diagnostics',np.sum(np.isnan(d['coupon_features_raw']),axis=0))
 for spec,A in [('score',g1[:,None]),('offset+score',np.c_[np.ones(nd),g1]),('free direction',np.c_[np.ones(nd),x])]:
  k,B=fit(A,full); err=(Y-A@B@design(k).T); oof=np.empty_like(err)
  for l in np.unique(lab):
   mask=lab!=l; kl,Bl=fit(A,mask); oof[~mask]=Y[~mask]-A[~mask]@Bl@design(kl).T
  def metr(e):return np.sqrt([np.mean(e[:,0]**2),np.mean(e[:,1:]**2),np.mean(e[:,-1]**2)])
  print(spec,'rate constant',round(k,5),'coefficients',B.round(5).tolist(),'full',metr(err).round(5),'LOLO',metr(oof).round(5),'worst lab',np.max([metr(oof[lab==l]) for l in np.unique(lab)],axis=0).round(5))
