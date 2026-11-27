import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svdvals
s=importlib.util.spec_from_file_location('arch','/tmp/hegel_continual_archive/alpha_m98_v2_c03_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; q=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']
def obs(q): return np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]
a,b=m._prepare(d); taus,th=m._fit(a,dt,t,obs(tr)); print('material',taus,th)
x=m._features(tr,d['test']); labs=np.r_[tr['lab_id'],d['test']['lab_id']]; ref=np.vstack([tr['ref_coupons'],d['test']['ref_coupons']]); xq=q['coupon_features_raw'].copy()
for g in np.unique(labs):
 r=ref[labs==g]; i=q['lab_id']==g; xq[i]=(xq[i]-np.median(r.mean(0)))/np.median(r.std(0))
xq=np.where(np.isnan(xq),x.mean(0),xq); off,w=m._calibration(d['calibration_load_cells']); cq=((q['past_channel_raw']-off)@w,(q['hold_channel_raw']-off)@w,np.c_[np.ones(len(xq)),xq]); y=obs(q)
def stat(e): return np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(abs(e[:,:-1])),np.median(abs(e[:,-1]))],8)
for name,e in [('train',m._design(a,dt,t,taus)@th-obs(tr)),('chamber baseline',m._design(cq,dt,t,taus)@th-y)]:print(name,stat(e))
def basis(tt):return np.vstack([np.c_[-np.expm1(-t[:,None]/tt),t],np.r_[1/tt,1]])
for tt in [taus]:
 B=basis(tt); coef=np.linalg.lstsq(B,y.T,rcond=None)[0].T; print('sv',svdvals(B),'free coefficients error',stat(coef@B.T-y)); base=m._design(cq,dt,np.array([1.]),taus); expected=np.c_[base[:,-1,:2]*taus*th[:2],cq[2]@th[2:]]; print('hold, features, fitted amp/drift, expected amp/drift, rate residual:'); print(np.round(np.c_[cq[1],xq,coef,expected,(coef@B.T-y)[:,-1]],6));print('raw q feature missing',np.isnan(q['coupon_features_raw']).sum(0))
 r=y-m._design(cq,dt,t,taus)@th; print('baseline residual rows'); print(np.round(r,5))
def var(logtau):
 B=basis(np.exp(logtau)); return (np.linalg.lstsq(B,y.T,rcond=None)[0].T@B.T-y).ravel()
for seed in [taus,taus*np.array([.7,1.4]),taus*1.5]:
 f=least_squares(var,np.log(seed),bounds=(np.log(.035),np.log(40)));print('free chamber taus',np.exp(f.x),stat(f.fun.reshape(y.shape)))
print('diagnostic field names',list(q))
