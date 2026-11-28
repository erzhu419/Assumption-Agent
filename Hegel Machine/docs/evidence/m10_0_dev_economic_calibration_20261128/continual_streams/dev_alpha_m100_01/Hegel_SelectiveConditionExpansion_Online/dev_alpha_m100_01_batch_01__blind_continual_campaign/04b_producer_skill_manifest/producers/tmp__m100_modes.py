import pickle,numpy as np
from scipy.optimize import minimize_scalar
from m100_base import _bridge,_loads,_features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=170)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=_features(tr,te); xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(q['lab_id']):]; b=_bridge(d['calibration_load_cells']); tau,theta=_fit(*_loads(tr,b),x[:n],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,h=_loads(q,b); C,R=_design(tau,p,h,xd,dt,t); ec=q['strain_curves_obs']-q['strain_0'][:,None]-C@theta; er=q['instant_rate_obs']-R@theta; E=np.c_[er,ec]
def temporal(k,rate=True):
 L=np.c_[-np.expm1(-t/k),t]; return np.vstack(([1/k,1],L)) if rate else L
def err(k,rate=True):
 L=temporal(k,rate); Y=E if rate else ec; return Y-(Y@np.linalg.pinv(L.T))@L.T
from scipy.optimize import least_squares
p0,h0=_loads(tr,b)
D=np.c_[q['instant_rate_obs'],q['strain_curves_obs']-q['strain_0'][:,None]]
parts=('lab_id','stratum','coupon_features_raw','ref_coupons')
xx={'pooled':xd,'train+q':_features(tr,q)[-len(h):],'test+q':_features(te,q)[-len(h):],'q-only':_features(q,q)[-len(h):]}
G=R[:,:2]*tau
Ls=lambda k:np.r_[1/k,-np.expm1(-t/k)]
linear=np.r_[1.,t]
F=np.c_[h,p,xd,G]
for name,Y,fix,nd in [('full',D,[],2),('full+fixed slow',D,[tau[1]],2),('res+fixed fast',E,[tau[0]],1),('res+fixed both',E,list(tau),1),('res-free',E,[],2)]:
 def calc(ks,ret=False):
  L=np.array([*(Ls(k) for k in np.r_[ks,fix]),linear]); A=Y@np.linalg.pinv(L); Z=Y-A@L; return (A,Z) if ret else Z.ravel()
 sols=[least_squares(calc,k,bounds=(.08,20),max_nfev=100) for k in ([[.5,4.3],[.6,1.4],[.5,1.0],[.3,3.],[.8,7.]] if nd==2 else [[.7],[1.3],[3.]])]; o=min(sols,key=lambda a:a.cost); A,Z=calc(o.x,True)
 print(name,'taus',np.r_[o.x,fix],'rms',np.sqrt(np.mean(Z**2,axis=0)),'coeff corr hold/p0..5/x0..3/gap0..1',np.corrcoef(np.c_[A,F].T)[:A.shape[1],A.shape[1]:].round(3))
 if name in ('full','res+fixed both'):
  print('rows A and last pair\n',np.c_[A,p[:,-2:]].round(4))
  for nm,xxv in xx.items():
   v=np.c_[np.ones(len(h)),xxv]; yc=A[:,-1]; co=np.linalg.lstsq(v,yc,rcond=None)[0]; print('drift',nm,'coef',co,'ratio',co/theta[2:],'rmse',np.mean((v@co-yc)**2)**.5)
  for j in range(A.shape[1]-1):
   M=np.c_[np.ones(len(h)),p,h,xd]; co=np.linalg.lstsq(M,A[:,j],rcond=None)[0]; print('linear A',j,co,'rms',np.mean((M@co-A[:,j])**2)**.5)
