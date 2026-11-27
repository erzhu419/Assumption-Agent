exec(open('/tmp/chamber_probe.py').read().split("print('parameters'")[0])
from scipy.optimize import least_squares, minimize_scalar
base=m._design(ch,dt,t,taus)@theta
mech=base-drift[:,None]*np.r_[t,1.]

def kernel(tau):
 return np.r_[-np.expm1(-t/tau),1/tau]

def profile(tau):
 q=kernel(tau); r=v-mech; a=r@q/(q@q); return r-a[:,None]*q,a
res=minimize_scalar(lambda tau: np.sum(profile(tau)[0]**2),bounds=(.3,2),method='bounded'); r,a=profile(res.x)
print('profile tau',res.x,'curve/rate RMSE',np.sqrt((r[:,:-1]**2).mean()),np.sqrt((r[:,-1]**2).mean()),'max',np.max(abs(r)))
print('lab, missing, amplitudes, drift, residual RMSE',np.round(np.c_[c['lab_id'],np.isnan(c['coupon_features_raw']).sum(1),a,drift,np.sqrt((r*r).mean(1))],4))
for name, groups in [('campaign',[(tr,te)]),('diagnostics',[(c,)]),('joint',[(tr,te,c)]),('train',[(tr,)]),('train+diag',[(tr,c)]),('test+diag',[(te,c)])]:
 g=groups[0]; xx=c['coupon_features_raw'].copy(); pp=np.vstack([z['ref_coupons'] for z in g]); ll=np.concatenate([z['lab_id'] for z in g]); cc=np.concatenate([z['stratum'] for z in g]); gx=np.vstack([z['coupon_features_raw'] for z in g]); _,ii,nn=np.unique(np.c_[ll,cc],axis=0,return_inverse=True,return_counts=True); ww=1/nn[ii];
 for j in np.unique(ll):
  p=pp[ll==j]; u=np.median(p.mean(0)); sig=np.median(p.std(0)); xx[c['lab_id']==j]=(xx[c['lab_id']==j]-u)/sig; gx[ll==j]=(gx[ll==j]-u)/sig
 ff=np.nansum(ww[:,None]*gx,axis=0)/(ww[:,None]*np.isfinite(gx)).sum(0); xx=np.where(np.isfinite(xx),xx,ff); dd=np.c_[np.ones(len(xx)),xx]@theta[2:]; fit=np.linalg.lstsq(np.c_[dd,np.ones(len(dd))],a,rcond=None)[0]; print(name,'gain/intercept',np.round(fit,4),'amplitude RMSE',np.sqrt(np.mean((np.c_[dd,np.ones(len(dd))]@fit-a)**2)))
print('singular values of chamber residual after stress',np.linalg.svd(v-mech,compute_uv=False).round(4))
print('assets',__import__('json').load(open('/holdout_task/public_asset_index.json'))['schema'].keys())
