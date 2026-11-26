import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import StratifiedKFold
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w01_batch_08_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
o=pickle.load(open('/holdout_task/public_alpha_w01_batch_08.pkl','rb'))[0]; d,g=o['dataset'],o['gt']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; x,xt=m._alignment(tr,te); maps=m._load_cells(d['calibration_load_cells']); inp=m._inputs(tr,x,maps,dt,d['yield_stress']); it=m._inputs(te,xt,maps,dt,d['yield_stress']); keep=np.isfinite(tr['coupon_features_raw']).all(1); ids=np.flatnonzero(keep); lab=tr['lab_id']; late=t>=np.median(t); rm=lambda z:np.sqrt(np.mean(z*z)); mad=lambda z:1.4826*np.median(abs(z-np.median(z)))
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; yt=g['te_clean']-te['strain_0'][:,None]; rt=g['te_true_rate0']
def fit(a):
 return m._fit({k:tr[k][a] for k in ('strain_curves_obs','strain_0','instant_rate_obs')},tuple(z[a] for z in inp),dt,t)
def candidates(a,z,labz,taus,w):
 c,_=m._design(taus,tuple(v[a] for v in inp),dt,t); res=np.median((y[a]-c@w)[:,late]/t[late],axis=1)
 f=x[a]@w[3:7]; fz=z@w[3:7]; out=np.zeros((3,len(z)))
 for j in range(4):
  k=lab[a]==j; kz=labz==j; scale=max(mad(res[k]),1e-4)
  for l,(A,B) in enumerate(((np.ones((k.sum(),1)),np.ones((kz.sum(),1))), (np.c_[np.ones(k.sum()),f[k]],np.c_[np.ones(kz.sum()),fz[kz]]), (np.c_[np.ones(k.sum()),x[a][k]],np.c_[np.ones(kz.sum()),z[kz]]))):
   p=least_squares(lambda p:(A@p-res[k])/scale,np.zeros(A.shape[1]),loss='soft_l1').x; out[l,kz]=B@p
   if len(a)==len(ids) and l==1: print('lab correction',j,p,'residual slope MAD',mad(res[k]),mad(res[k]-A@p))
 return np.vstack((np.zeros(len(z)),out))
ecv=np.zeros((4,len(y),len(t))); rcv=np.zeros((4,len(y)))
for A,B in StratifiedKFold(5,shuffle=True,random_state=183).split(ids,3*lab[ids]+tr['stratum'][ids]):
 a,b=ids[A],ids[B]; tau,w=fit(a); c,r=m._design(tau,tuple(v[b] for v in inp),dt,t); delta=candidates(a,x[b],lab[b],tau,w); ecv[:,b]=c@w-y[b]+delta[:,:,None]*t; rcv[:,b]=r@w-tr['instant_rate_obs'][b]+delta
b=np.isfinite(te['coupon_features_raw']).all(1); tau,w=fit(ids); c,r=m._design(tau,it,dt,t); delta=candidates(ids,xt,te['lab_id'],tau,w); et=c@w-yt+delta[:,:,None]*t; er=r@w-rt+delta
for i in range(4):
 print(('baseline','intercept','affine feature score','unrestricted slopes')[i]);
 for j in range(4):
  k=keep&(lab==j); q=b&(te['lab_id']==j); ec=ecv[i,k]; print(j,'OOF late drift MAD, trimmed curve RMSE, rate RMSE',np.round([mad(np.median(ec[:,late]/t[late],axis=1)),rm(np.clip(ec,-.07,.07)),rm(rcv[i,k])],6),'public complete curve/rate RMSE',np.round([rm(et[i,q]),rm(er[i,q])],6))
