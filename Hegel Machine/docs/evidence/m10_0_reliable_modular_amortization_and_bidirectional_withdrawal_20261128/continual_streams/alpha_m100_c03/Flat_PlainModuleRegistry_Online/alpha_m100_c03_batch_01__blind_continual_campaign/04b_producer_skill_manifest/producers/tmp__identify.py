import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from joint import *
xx=di['coupon_features_raw']; combos=[('pool',(tr,te)),('all',(tr,te,di)),('diag',(di,)),('train',(tr,))]; variants={}
for name,ss in combos:
 rr=np.vstack([s['ref_coupons'] for s in ss]); ll=np.concatenate([s['lab_id'] for s in ss]); v=xx.copy()
 for lab in np.unique(ll):
  a=rr[ll==lab]; m=di['lab_id']==lab; v[m]=(v[m]-np.median(a.mean(0)))/np.median(a.std(0))
 variants[name]=np.where(np.isnan(v),Xd,v)
print('compare drift variations on complete',np.c_[dr,*[theta[2]+v@theta[3:] for v in variants.values()]][~np.isnan(xx).any(1)][:10]);
for name,v in variants.items():
 d0=theta[2]+v@theta[3:]; Z=mat(one,h,*p.T,d0); Zx=mat(one,h,*p.T,*v.T);print('\nNORM',name)
 for nm,C in [('dr-sparse',[mat(one,d0),mat(one,d0,z[:,1]),mat(one,d0,h)]),('fullhist',[Z,Z,Z]),('unconstrained',[Zx,Zx,Zx]),('equil',[mat(one,d0),mat(one,d0),mat(one,d0)])]:
  good=~np.isnan(xx).any(1);yp,w=fitlaw(C,good); e=y-yp; pe=np.vstack([y[k]-fitlaw(C,good & (np.arange(len(h))!=k))[0][k] for k in np.flatnonzero(good)]); print(nm,'train',np.round(np.sqrt(np.mean(e[good]**2,0))[[0,5,11,12]],4),'LOO',np.round(np.sqrt(np.mean(pe**2,0))[[0,5,11,12]],4),'W',w)
 print('per-coupon projected coefficients vs zero-memory',np.c_[q-base,d0][good][:10])
print('diagnostic experimental calibration sweeps keys',d['calibration_load_cells'].keys())
