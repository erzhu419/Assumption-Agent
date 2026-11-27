import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c04_s00_solver as s
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; dg=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; br=d['calibration_load_cells']; raw=br['bridge_raw']; true=br['true_channel_loads']; b=s._bridge(br)
xb=s._features(tr,te); x=xb[:len(tr['lab_id'])]; p,h=s._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,theta=s._fit(x,p,h,dt,t,y,r)
print('tau',tau,'theta',theta); print('linear bridge rms, max by channel',np.sqrt(np.mean((np.c_[np.ones(len(raw)),raw]@b-true)**2,0)),np.max(np.abs(np.c_[np.ones(len(raw)),raw]@b-true),0))
for k in br['single_channel_sweeps']: print('sweep',k,np.shape(br['single_channel_sweeps'][k]),np.asarray(br['single_channel_sweeps'][k]).ravel()[:10])
for name in ['ref_coupons','lab_id','stratum','coupon_features_raw']:
 pass
joint={k:np.concatenate([te[k],dg[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}; xd=s._features(tr,joint)[-len(dg['lab_id']):]
from scipy.optimize import minimize_scalar
p,h=s._loads(dg,b); Y=dg['strain_curves_obs']-dg['strain_0'][:,None]; R=dg['instant_rate_obs']; A=np.c_[-np.expm1(-t[:,None]/tau),t]; B=np.c_[np.r_[A[:,0],1/tau[0]],np.r_[A[:,1],1/tau[1]],np.r_[t,1.]]; obs=np.c_[Y,R];
lab=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xnew=dg['coupon_features_raw'].copy(); st=np.r_[tr['stratum'],te['stratum']]; counts=np.unique(np.c_[lab,st],axis=0,return_inverse=True,return_counts=True); w=1./counts[2][counts[1],None]; rawx=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); fill=np.sum(w*xb*(~np.isnan(rawx)),0)/np.sum(w*(~np.isnan(rawx)),0)
for g in np.unique(lab):
 panel=refs[lab==g]; xnew[dg['lab_id']==g]=(xnew[dg['lab_id']==g]-np.median(panel.mean(0)))/np.median(panel.std(0))
xnew=np.where(np.isnan(xnew),fill,xnew)
print('max diag feature coordinate difference',np.max(abs(xnew-xd),0))
p,h=s._loads(dg,b); n=len(h); later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; m=-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau); z=p@m; gap=h[:,None]-z; print('dt, t',dt,t); print('diag keys',list(dg)); print('raw and calibrated hold ranges',h.min(),h.max())
D=B[:,:2]; target=obs-(gap*theta[:2])@D.T
Z=lambda k:np.r_[-k*np.expm1(-t/k),1.]
from scipy.optimize import minimize_scalar
from scipy.special import expit
lab=dg['lab_id']; dr=theta[2]+xnew@theta[3:]; last=p[:,-1]; mask=np.isfinite(dg['coupon_features_raw']).all(1)
def assess(drift):
 candidates={'unit':[], 'sigmoid':[drift,expit(last)-.5], 'tanh':[drift,np.tanh(last)],'drift_tanh':[drift,np.tanh(last),drift*np.tanh(last)],'recent_step':[drift,np.tanh(h-last)],'recent_delta':[drift,np.tanh(last-p[:,-2])], 'tanh_mem':[drift,np.tanh(z[:,0])], 'tanh_mean':[drift,np.tanh(p.mean(1))], 'rational':[drift,last/(1+last**2)], 'sin':[drift,np.sin(last)], 'softsign':[drift,last/(1+np.abs(last))], 'sigmoid_pair':[drift,expit(last)-.5,expit(p[:,-2])-.5], 'scaled_drift':[drift,drift*expit(last)]}
 for name, cols in candidates.items():
  F=np.column_stack([np.ones(n),*cols]) if cols else np.empty((n,0)); R=.01*np.eye(F.shape[1]);
  def fit(m):
   def trial(k, ret=False):
    q=Z(k); a=target@q/(q@q)-drift; C=np.linalg.solve(F[m].T@F[m]+R,F[m].T@a[m]); e=target-(drift+F@C)[:,None]*q
    return (e,C) if ret else np.mean(e[m]**2)
   k=minimize_scalar(trial,bounds=(.85,1.15),method='bounded').x if cols else 1.; return k,*trial(k,True)
  def score(groups):
   e=np.empty_like(target)
   for g in np.unique(groups):
    m=groups==g; e[m]=fit(~m)[1][m]
   return [max(np.sqrt(np.mean(e[lab==g]**2)) for g in np.unique(lab)),np.percentile(abs(e[:,:-1]),90),np.percentile(abs(e[:,-1]),90)]
  k,e,C=fit(np.ones(n,bool)); print(name,'LOLO',np.round(score(lab),4),'LOCO',np.round(score(np.arange(n)),4),'fit',np.round([k,np.sqrt(np.mean(e**2)),*C],5))
assess(dr)
print('calibrated deployment last stress quantiles',np.quantile(s._loads(te,b)[0][:,-1],[0,.25,.5,.75,1])); print('diag',np.quantile(last,[0,.25,.5,.75,1])); print('dep feature missing',np.isnan(te['coupon_features_raw']).sum(0))
