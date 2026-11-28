import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m100_c04_s00_solver as s
np.set_printoptions(precision=5,suppress=True)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; p=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; cal=s._calibration(c); n=len(tr['lab_id']); x=s._features(tr,te); ti=s._inputs(tr,x[:n],cal); tau,theta=s._fit(ti,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau theta cal',tau,theta,cal)
r=c['bridge_raw']; loads=c['true_channel_loads']; A=np.c_[r,np.ones(len(r))]; Q=np.c_[A,*[r[:,k]*r[:,j] for k in range(4) for j in range(k,4)]]
for mat in [A,Q]:
 err=(mat@np.linalg.lstsq(mat,loads,rcond=None)[0]-loads)/(1-np.sum(np.linalg.qr(mat,mode='reduced')[0]**2,axis=1))[:,None]; print('bridge LOO RMSE chan and sum',np.sqrt(np.mean(err**2,axis=0)),np.sqrt(np.mean(err.sum(1)**2)))
print('cmd sweeps by lab slopes', [np.polyfit(c['single_channel_sweeps']['cmd_stress'][c['single_channel_sweeps']['lab_id']==j],c['single_channel_sweeps']['true_stress'][c['single_channel_sweeps']['lab_id']==j],1) for j in np.unique(c['single_channel_sweeps']['lab_id'])]);
px=s._features(tr,p)[n:]; past,h,cov=s._inputs(p,px,cal); a,b=s._design(tau,(past,h,cov),dt,t); Y=np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]; Y0=np.c_[a@theta,b@theta]
def basis(ta): return np.r_[np.c_[-np.expm1(-t[:,None]/ta),t],[[1/ta[0],1/ta[1],1]]]
def temporal(ta):
 D=basis(ta); return Y.T-D@np.linalg.lstsq(D,Y.T,rcond=None)[0]
ta=least_squares(lambda z:temporal(z).ravel(),tau,bounds=(.05,25)).x; print('diagnostic temporal fit',ta,'rmse old/new',*[np.sqrt(np.mean(temporal(z)**2)) for z in [tau,ta]])
D=basis(tau); V=np.linalg.lstsq(D,(Y-Y0).T,rcond=None)[0].T; Yb=Y-(cov@theta[2:])[:,None]*np.r_[t,1]; gains=np.linalg.lstsq(D,Yb.T,rcond=None)[0].T; print('residual SVD',np.linalg.svd(Y-Y0,compute_uv=False)[:8]); print('baseline rmse',np.sqrt(np.mean((Y-Y0)**2,axis=0))); print('diagnostic source ranges',np.min(past,axis=0),np.max(past,axis=0)); print('test source ranges',*[np.round(getattr(np,k)(s._inputs(te,x[n:],cal)[0],axis=0),3) for k in ['min','max']]);
def panel_x(p):
    raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); labs=np.r_[tr['lab_id'],te['lab_id']]; strata=np.r_[tr['stratum'],te['stratum']]; panel=np.vstack([tr['ref_coupons'],te['ref_coupons']]); z=p['coupon_features_raw'].copy();
    for lab in np.unique(labs):
        m=labs==lab; means=np.nanmedian(np.nanmean(panel[m],axis=0)); sd=np.nanmedian(np.nanstd(panel[m],axis=0)); idx=p['lab_id']==lab; z[idx]=(z[idx]-means)/sd; raw[m]=(raw[m]-means)/sd
    _,i,cnt=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True); w=1/cnt[i,None]; mu=np.nansum(w*raw,axis=0)/np.sum(w*np.isfinite(raw),axis=0); return np.where(np.isnan(z),mu,z)
M=np.isfinite(p['coupon_features_raw']).all(1); print('complete',M.sum(),'by lab',np.bincount(p['lab_id'][M])); print('missing rows',np.where(~M)[0]); print('material',tau,theta)
for name,xx in [('train-diag',px),('train-test ref',panel_x(p)),('diagnostic only',s._features(p,p)[:len(h)]),('three panels',s._features(tr,{k:np.concatenate([te[k],p[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')})[-len(h):])]:
 for maskname,mask in [('complete',M),('all',np.ones(len(h),bool))]:
  drift=theta[2]+xx@theta[3:]; Z=np.c_[np.ones(len(h)),h,drift]; O=np.outer(h,D[:,:2]@theta[:2]); y=Y-O
  for label,F,offset in [('free response',np.einsum('ij,k->ikj',Z,np.ones(len(D))),0),('drift signal',np.einsum('ij,kl->ikjl',Z[:,[0,2]],D).reshape(len(h),len(D),-1),O),('with hold calibration',np.einsum('ij,kl->ikjl',Z,D).reshape(len(h),len(D),-1),0)]:
   yc=Y-offset; coef=np.linalg.lstsq(F[mask].reshape(-1,F.shape[-1]),yc[mask].ravel(),rcond=None)[0]; er=F@coef-yc; rms=lambda e: np.r_[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.max(np.sqrt(np.mean(e[:,:-1]**2,axis=1)))]; worst=[]
   for lab in np.unique(p['lab_id']):
    m=mask & (p['lab_id']!=lab); v=mask & (p['lab_id']==lab); q=np.linalg.lstsq(F[m].reshape(-1,F.shape[-1]),yc[m].ravel(),rcond=None)[0]; worst.append(rms(F[v]@q-yc[v]))
   if label!='free response': print(name,maskname,label,'fit',rms(er[mask]),'worst',np.max(worst,axis=0),'coefs',coef,'missing residuals',np.round(er[~M][:,[-2,-1]],4))
  B=np.linalg.lstsq(np.c_[np.ones(mask.sum()),h[mask],xx[mask]],Y[mask]@np.linalg.pinv(D.T),rcond=None)[0]; print('feature coef ratios',np.round(B[2:]/theta[3:,None],3))
