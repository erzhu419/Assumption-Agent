import sys
sys.path.insert(0,'/holdout_task')
import json,pickle,numpy as np
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
from chamber_probe_base import _features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=145)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=a['dataset']; p=json.load(open('/holdout_task/_upstream_param_payload.json')); d=calibrate_upstream(raw,p); da=calibrate_upstream(raw,None)
print('dynamic bridge deltas',[(k,np.max(abs(d['train'][k]-da['train'][k]))) for k in ('past_channel_raw','hold_channel_raw')]); print('payload',p)
q=calibrate_upstream({**raw,'test':a['campaign_diagnostics']},p)['test']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; xtr,xte=_features(tr,te)
def stress(s): return [s[k].sum(-1) for k in ('past_channel_raw','hold_channel_raw')]
def augment(x): return np.c_[np.ones(len(x)),x]
def observations(s): return np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]
y=observations(tr); taus,b=_fit(*stress(tr),augment(xtr),dt,t,y[:,:-1],y[:,-1]); print('intrinsic',taus,b)
def intrinsic(split,x):
 c,r=_design(taus,*stress(split),augment(x),dt,t);return np.c_[c@b,r@b]
print('benchtop residual rms',np.sqrt(np.mean((y-intrinsic(tr,xtr))**2,axis=0))); print('benchtop residual MAD',np.median(abs(y-intrinsic(tr,xtr)),axis=0))
x=np.array(q['coupon_features_raw']); labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xx=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); sx=x.copy()
for lab in np.unique(labs):
 panel=refs[labs==lab]; center=np.median(panel.mean(0)); scale=np.median(panel.std(0)); x[q['lab_id']==lab]=(x[q['lab_id']==lab]-center)/scale; xx[labs==lab]=(xx[labs==lab]-center)/scale; panel=q['ref_coupons'][q['lab_id']==lab]; sx[q['lab_id']==lab]=(sx[q['lab_id']==lab]-np.median(panel.mean(0)))/np.median(panel.std(0))
_,ci,cc=np.unique(np.c_[labs,np.r_[tr['stratum'],te['stratum']]],axis=0,return_counts=True,return_inverse=True); w=1/cc[ci,None]; imp=np.nansum(w*xx,0)/(w*~np.isnan(xx)).sum(0)
past,h=stress(q); age=np.cumsum(dt[::-1])[::-1]-dt; z=past@(np.exp(-age[:,None]/taus)*-np.expm1(-dt[:,None]/taus)); obs=observations(q); phi=np.c_[-np.expm1(-t[:,None]/taus),t]; phi=np.r_[phi, [np.r_[1/taus,1]]]; archive=lambda x:np.c_[h[:,None]*b[:2]+(augment(x)@b[2:])[:,None]*[.19552,1.90306],.01715*(augment(x)@b[2:])]@phi.T
for tag,x0 in [('deployment',x),('panel',sx),('train+diag',_features(tr,q)[1])]:
 x0=np.where(np.isnan(x0),imp,x0); drift=augment(x0)@b[2:]; pred=intrinsic(q,x0); ds=obs-pred
 good=np.isfinite(x).all(1); print('\ncoord',tag,'N complete',good.sum(),'baseline & archived curve/rate RMSE',[(np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))) for e in (ds,obs-archive(x0))]); print('SVD residual all / complete',np.linalg.svd(ds,compute_uv=False)[:7],np.linalg.svd(ds[good],compute_uv=False)[:7]); print('resid time median',np.median(ds,0))
 reset=np.c_[h[:,None]*b[:2],np.zeros(len(h))]@phi.T
 blocks={'drift':drift[:,None],'hold':h[:,None],'z':z,'const':np.ones((len(h),1)),'X':x0,'last':past[:,-1:],'past':past}
 for terms in [('drift',),('drift','z'),('drift','hold','z','const'),('drift','hold','z','X','const'),('drift','hold','past','X','const')]:
  A=np.column_stack([blocks[k] for k in terms]); M=(A[:,:,None,None]*phi[None,None,:,:].transpose(0,1,3,2)).reshape(len(h),-1,len(t)+1).transpose(0,2,1); base=reset; nm=M.shape[-1]; yy=obs-base
  def fit(g):
   ma,ya=M[g].reshape(-1,nm),yy[g].ravel(); init=np.linalg.lstsq(ma,ya,rcond=1e-9)[0]; return least_squares(lambda v:ma@v-ya,init,loss='soft_l1',f_scale=.015).x
  coef=fit(np.arange(len(h))); err=yy-M@coef; cv=np.array([yy[i]-M[i]@fit(np.arange(len(h))!=i) for i in range(len(h))]); print(terms,'rms',np.round([np.sqrt(np.mean(err[:,:-1]**2)),np.sqrt(np.mean(err[:,-1]**2)),np.sqrt(np.mean(cv[:,:-1]**2)),np.sqrt(np.mean(cv[:,-1]**2))],5),'coef', np.round(coef.reshape(-1,3),4).tolist() if len(terms)<=4 else '', 'MAD',np.median(abs(err)))
