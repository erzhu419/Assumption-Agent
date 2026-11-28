import pickle,numpy as np
from scipy.optimize import minimize_scalar
from m100_base import _bridge,_loads,_features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=170)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=_features(tr,te); xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(q['lab_id']):]; b=_bridge(d['calibration_load_cells']); tau,theta=_fit(*_loads(tr,b),x[:n],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,h=_loads(q,b); C,R=_design(tau,p,h,xd,dt,t); ec=q['strain_curves_obs']-q['strain_0'][:,None]-C@theta; er=q['instant_rate_obs']-R@theta; E=np.c_[er,ec]
def temporal(k,rate=True):
 L=np.c_[-np.expm1(-t/k),t]; return np.vstack(([1/k,1],L)) if rate else L
def err(k,rate=True):
 L=temporal(k,rate); Y=E if rate else ec; return Y-(Y@np.linalg.pinv(L.T))@L.T
for rate in (True,False):
 opt=minimize_scalar(lambda k: np.sum(err(k,rate)**2),bounds=(.1,10),method='bounded'); L=temporal(opt.x,rate); Y=E if rate else ec; AB=Y@np.linalg.pinv(L.T); print('tau',opt.x,'RMS',np.sqrt(np.mean(err(opt.x,rate)**2)), 'initial RMS',np.sqrt(np.mean((AB@np.array([1/opt.x,1])-er)**2))); print('fixed times',[(k,np.sqrt(np.mean(err(k,rate)**2))) for k in (.45,.55,.58,.6,.65,.7,.75,.8,.85,.9,1.,1.1,1.2,1.3,1.4,1.5,2.)]);
 if rate: A,B=AB.T
print('lab str A B hold p[0:6] x[0:4]');print(np.c_[q['lab_id'],q['stratum'],A,B,h,p,xd]); print('feature correlations order hold,p0:6,x0:4,gap0:2,stratum'); F=np.c_[h,p,xd,R[:,:2]*tau,q['stratum']]; print(np.corrcoef(np.c_[A,B,F].T)[:2,2:]); print('sweeps',d['calibration_load_cells']['single_channel_sweeps'])
