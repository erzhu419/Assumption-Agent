import pickle,numpy as np
from scipy.optimize import minimize_scalar
from m100_base import _bridge,_loads,_features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=170)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=_features(tr,te); xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(q['lab_id']):]; b=_bridge(d['calibration_load_cells']); tau,theta=_fit(*_loads(tr,b),x[:n],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,h=_loads(q,b); C,R=_design(tau,p,h,xd,dt,t); ec=q['strain_curves_obs']-q['strain_0'][:,None]-C@theta; er=q['instant_rate_obs']-R@theta; E=np.c_[er,ec]
def temporal(k,rate=True):
 L=np.c_[-np.expm1(-t/k),t]; return np.vstack(([1/k,1],L)) if rate else L
def err(k,rate=True):
 L=temporal(k,rate); Y=E if rate else ec; return Y-(Y@np.linalg.pinv(L.T))@L.T
from scipy.optimize import minimize_scalar
print('sizes',len(tr['lab_id']),len(te['lab_id']),len(q['lab_id']))
Xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(h):]
for name, parts in [('tr',(tr,)),('te',(te,)),('q',(q,)),('trte',(tr,te))]:
 l=np.concatenate([v['lab_id'] for v in parts]); rr=np.concatenate([v['ref_coupons'] for v in parts]); print(name,[(round(np.median(rr[l==j].mean(0)),3),round(np.median(rr[l==j].std(0)),3)) for j in range(4)])
for mode in ('new','shift','extra'):
 a0=R[:,0]*tau[0]*theta[0]; L0=np.r_[1/tau[0],-np.expm1(-t/tau[0])]; Y=E+(a0[:,None]*L0 if mode=='shift' else 0)
 def tm(k):
  L=temporal(k)
  return np.c_[L,L0] if mode=='extra' else L
 def calc(k):
  L=tm(k); co=Y@np.linalg.pinv(L.T); return np.sum((Y-co@L.T)**2),co
 opt=minimize_scalar(lambda k:calc(k)[0],bounds=(.36,1.2),method='bounded'); v,co=calc(opt.x); print(mode,opt.x,np.sqrt(v/Y.size));
 if mode=='extra': print('extra amplitude corr/fit to old fast',np.corrcoef(co[:,2],a0)[0,1],np.linalg.lstsq(np.c_[np.ones(len(h)),a0],co[:,2],rcond=None)[0])
 if mode=='new':
  A,B=co.T; order=np.argsort(p[:,-1]); print('last stress, A,B, hold,prev,lab,str,raw missing'); print(np.c_[p[:,-1],A,B,h,p[:,-2],q['lab_id'],q['stratum'],np.isnan(q['coupon_features_raw']).sum(1)][order]);
  for name,F in [('features',np.c_[xd]),('features+stress',np.c_[xd,h,p[:,-1]]),('features+lab',np.c_[xd,np.eye(4)[q['lab_id']][:,1:]]),('features+str',np.c_[xd,np.eye(3)[q['stratum']][:,1:]])]:
   M=np.c_[np.ones(len(h)),F]; coef=np.linalg.lstsq(M,B,rcond=None)[0]; print('B',name,'coef',coef,'rmse',np.sqrt(np.mean((B-M@coef)**2)))
