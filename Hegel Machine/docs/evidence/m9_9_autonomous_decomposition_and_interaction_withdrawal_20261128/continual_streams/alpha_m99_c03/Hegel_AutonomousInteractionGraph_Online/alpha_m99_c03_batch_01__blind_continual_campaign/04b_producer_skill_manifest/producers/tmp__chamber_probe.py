import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m99_c03_s00_solver as m
np.set_printoptions(precision=5,suppress=True,linewidth=130)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=it['dataset']; tr,te,q=d['train'],d['test'],it['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; br=m._calibration(cal)
r=cal['bridge_raw']; y=cal['true_channel_loads']; e=np.c_[np.ones(len(r)),r]@br-y
print('bridge',r.shape,'range',np.min(r,0),np.max(r,0),'channel and total residual RMSE',np.sqrt(np.mean(e**2,0)),np.std(e.sum(1))); print('bridge coefficients',br); s=cal['single_channel_sweeps']; print('sweeps',type(s),list(s)[:4] if isinstance(s,dict) else np.shape(s))
x=m._features(tr,te); a=m._coupon_inputs(tr,x[:len(tr['lab_id'])],br); taus,th=m._fit(a,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',taus,th)
all_lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xq=q['coupon_features_raw'].copy(); print('diagnostic missing',np.isnan(xq).sum(0));
for l in np.unique(all_lab):
 p=ref[all_lab==l]; print('panel',l,np.median(p.mean(0)),np.median(p.std(0))); xq[q['lab_id']==l]=(xq[q['lab_id']==l]-np.median(p.mean(0)))/np.median(p.std(0))
_,cid,cnt=np.unique(np.c_[all_lab,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); w=1/cnt[cid]; xq=np.where(np.isnan(xq),np.average(x,axis=0,weights=w),xq)
qa=m._coupon_inputs(q,xq,br)
def summarize(b,inp):
 c,v=m._design(taus,inp,dt,t); yc=b['strain_curves_obs']-b['strain_0'][:,None]; yr=b['instant_rate_obs']; ec=yc-c@th; er=yr-v@th; print('baseline curve/rate quantiles',np.quantile(abs(ec),[.5,.9,.95,1]),np.quantile(abs(er),[.5,.9,.95,1])); return ec,er,yc,yr
summarize(tr,a); ec,er,yc,yr=summarize(q,qa)
F=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1.]]); print('design singular values',np.linalg.svd(F,compute_uv=False)); print('diagnostic residual first, middle, end, rate'); print(np.c_[ec[:,0],ec[:,5],ec[:,-1],er]); print('residual SVs',np.linalg.svd(ec,compute_uv=False)); print('diagnostic input stress first records',np.c_[qa[0],qa[1]][:5]); print('diagnostic latent corrections and fit medians'); corr=[]
for yy in np.c_[ec,er]:
 p=least_squares(lambda p:F@p-yy,np.linalg.lstsq(F,yy,rcond=None)[0],loss='soft_l1',f_scale=.008).x; corr.append(p); print(np.r_[p,np.median(abs(F@p-yy))]);
np.savez('/tmp/chamber_probe.npz',taus=taus,theta=th,past=qa[0],hold=qa[1],features=xq,curve=yc,rate=yr,ec=ec,er=er,corr=corr,t=t,dt=dt,bridge=br)
