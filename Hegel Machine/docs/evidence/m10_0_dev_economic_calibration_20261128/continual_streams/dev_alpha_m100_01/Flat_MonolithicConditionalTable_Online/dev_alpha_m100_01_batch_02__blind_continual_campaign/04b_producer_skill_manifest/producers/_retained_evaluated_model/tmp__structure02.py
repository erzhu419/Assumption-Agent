import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from explore02 import *
base=R@th; gap=C[:,-1,:2]/(-np.expm1(-t[-1]/tau)); aux=a['aux_chamber_telemetry']; n=len(hc); ones=np.ones(n)
print('dr source: missing rows, labs, strata',np.flatnonzero(np.isnan(c['coupon_features_raw']).any(1)),c['lab_id'],c['stratum']); print('axis t plus rate',np.r_[t,0]);
mat=np.c_[ones,dr,hc]; q=np.linalg.lstsq(mat,mall,rcond=None)[0]; resid=mall-mat@q
print('time coefficients rows intercept, drift, hold',q); print('residual top direction', np.linalg.svd(resid,full_matrices=False)[2][:2]); print('residual SVD',np.linalg.svd(resid,compute_uv=False)); print('residual coupon coordinates and load waveform slopes',np.c_[resid[:,[-2,-1]],gap,pc[:,-2:]])
for key,f in {'x':xc,'gap':gap,'lab':np.eye(1+max(c['lab_id']))[c['lab_id']][:,1:],'stratum':np.unique(c['stratum'],return_inverse=True)[1][:,None], 'past':pc,'hold_past':hc[:,None]-pc,'rate':base[:,None],'gaps_by_drift':gap*dr[:,None],'load_by_drift':hc[:,None]*dr[:,None],'missing':np.isnan(c['coupon_features_raw'])[:,[0,3]]}.items():
 f=np.asarray(f); A=np.c_[mat,f]; pinv=np.linalg.pinv(A); rm=mall-A@(pinv@mall); lev=np.einsum('ij,ji->i',A,pinv); loo=rm/np.maximum(1e-9,1-lev[:,None]); print(key,'rms late/rate',rms(rm)[[-2,-1]],'loo late/rate',rms(loo)[[-2,-1]],'residual correlations',np.corrcoef(np.c_[resid[:,[-2,-1]],f].T)[:2,2:])
for p0 in (a,b,c):
 print('labs anchors',[(l,np.round(p0['ref_coupons'][p0['lab_id']==l].mean(0),3)) for l in np.unique(p0['lab_id'])]);
