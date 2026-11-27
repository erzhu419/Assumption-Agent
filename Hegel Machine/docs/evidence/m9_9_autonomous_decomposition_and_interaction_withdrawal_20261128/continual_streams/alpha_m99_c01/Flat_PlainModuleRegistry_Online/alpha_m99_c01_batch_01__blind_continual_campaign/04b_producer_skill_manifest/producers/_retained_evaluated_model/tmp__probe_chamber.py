import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m99_c01_s00_solver.py'); base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr=ds['train']; te=ds['test']; d=item['campaign_diagnostics']; t=ds['eval_times']; dt=ds['past_dt']; cal=ds['calibration_load_cells']
raw=cal['bridge_raw']; loads=cal['true_channel_loads']; B=np.c_[np.ones(len(raw)),raw]; E=loads-B@np.linalg.lstsq(B,loads,rcond=None)[0]; print('bridge affine residual rms',np.sqrt(np.mean(E**2,axis=0)),'total',np.std(E.sum(1)))
w=base._bridge(cal); x,xt=base._features(tr,te); p,h=base._loads(tr,w); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; tau,beta=base._fit(x,p,h,dt,t,y,rate); print('tau beta',tau,beta)
xc=d['coupon_features_raw'].copy(); lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.vstack((tr['ref_coupons'],te['ref_coupons']));
for l in np.unique(lab):
 r=ref[lab==l]; xc[d['lab_id']==l]=(xc[d['lab_id']==l]-np.median(r.mean(0)))/np.median(r.std(0))
_,inv,cts=np.unique(np.c_[lab,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); xc=np.where(np.isfinite(xc),xc,np.average(np.vstack((x,xt)),axis=0,weights=1/cts[inv])); pc,hc=base._loads(d,w); yc=d['strain_curves_obs']-d['strain_0'][:,None]; rc=d['instant_rate_obs'];
for tag,xx,pp,hh,yy,rr in [('train',x,p,h,y,rate),('chamber',xc,pc,hc,yc,rc)]:
 a,b=base._design(tau,xx,pp,hh,dt,t); print(tag,'baseline median abs/max',np.median(abs(a@beta-yy)),np.max(abs(a@beta-yy)),np.median(abs(b@beta-rr)),np.max(abs(b@beta-rr)))
obs=np.c_[yc,rc]; times=np.r_[t,0.];
def fit(lam):
 u=np.exp(lam); A=np.c_[np.r_[-np.expm1(-t[:,None]/u),[1/u]],np.r_[t,1.]]; coeff=np.linalg.lstsq(A,obs.T,rcond=None)[0].T; return coeff,coeff@A.T-obs
for guess in [tau,[.2,2],[.9,5]]:
 sol=least_squares(lambda v:fit(v)[1].ravel(),np.log(guess),loss='soft_l1',f_scale=.005,max_nfev=150); coeff,res=fit(sol.x); print('chamber free taus',np.exp(sol.x),'abs quantiles',np.quantile(abs(res),[.5,.9,1]))
coeff,res=fit(np.log(tau)); print('fixed taus abs residual per coupon',np.round(np.median(abs(res),axis=1),4)); a,b=base._design(tau,xc,pc,hc,dt,t); nom=np.c_[b[:,:2]*tau*beta[:2],b[:,2:]@beta[2:]]; delta=coeff-nom; print('nom and delta:');print(np.round(np.c_[nom,delta],4)); F=np.c_[hc,pc,xc]; print('corr delta with hold,past,features',np.round(np.corrcoef(F.T,delta.T)[:-3,-3:],2));
