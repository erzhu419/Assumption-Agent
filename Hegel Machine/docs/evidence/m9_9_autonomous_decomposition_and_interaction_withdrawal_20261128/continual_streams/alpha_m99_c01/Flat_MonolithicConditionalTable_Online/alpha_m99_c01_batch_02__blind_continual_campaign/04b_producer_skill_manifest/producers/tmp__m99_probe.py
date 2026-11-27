import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_m99_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,c=d['train'],d['test'],i['campaign_diagnostics']; t,dt=d['eval_times'],d['past_dt']; X,Xe=m._features(tr,te); br=m._bridge(d['calibration_load_cells']); p,h=m._loads(tr,br); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,beta=m._fit(X,p,h,dt,t,y,r)
# Apply the deployed feature transform to diagnostic rows, without letting them set it.
xc=c['coupon_features_raw'].copy(); labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); xx=np.vstack([X,Xe]); impute=np.array([xx[~np.isfinite(raw[:,j]),j][0] for j in range(4)])
for l in np.unique(labs):
 rr=refs[labs==l]; sel=c['lab_id']==l; xc[sel]=(xc[sel]-np.median(rr.mean(0)))/np.median(rr.std(0))
xc=np.where(np.isfinite(xc),xc,impute); x_joint,xc_joint=m._features({k:np.concatenate((tr[k],te[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')},c)
pc,hc=m._loads(c,br); yc=c['strain_curves_obs']-c['strain_0'][:,None]; rc=c['instant_rate_obs']; A,B=m._design(tau,X,p,h,dt,t); ac,bc=m._design(tau,xc,pc,hc,dt,t); cy=ac@beta; cr=bc@beta
if __name__=='__main__':
 print('tau,beta',tau,beta)
 def fmt(e):return np.round(np.quantile(abs(e),[.5,.8,.95,1]),6)
 print('absolute errors train (curve,rate)',fmt(A@beta-y),fmt(B@beta-r)); print('absolute errors chamber baseline',fmt(cy-yc),fmt(cr-rc));
 if hasattr(m,'_chamber_response'):
  yp,rp=m._chamber_response(tau,beta,xc,hc,t); print('absolute errors chamber prior',fmt(yp-yc),fmt(rp-rc))
 print('curve svd raw',np.linalg.svd(yc,compute_uv=False)); print('curve svd difference',np.linalg.svd(yc-cy,compute_uv=False));
 E=np.column_stack((-np.expm1(-t[:,None]/tau),t)); coef=np.linalg.lstsq(E,yc.T,rcond=None)[0].T; print('curve temporal basis error',fmt(coef@E.T-yc),'rate tie error',fmt(coef@np.r_[1/tau,1]-rc));
 base=np.column_stack((bc[:,:2]*tau*beta[:2],bc[:,2:]@beta[2:])); print('observed minus baseline by coeff, first rows:\n',np.round((coef-base)[:14],4)); print('fit each response on linear load and feature inputs:')
 F=np.c_[np.ones(len(hc)),hc,pc,xc]; fit=np.linalg.lstsq(F,coef,rcond=None)[0]; print(np.round(fit,5));print('linear regression errors',fmt(F@fit-coef)); print('load feature ranges',np.quantile(np.c_[hc,pc,xc],[0,.5,1],axis=0)); print('diagnostic feature transforms differ',np.max(abs(xc-xc_joint),axis=0));
