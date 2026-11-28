import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/chamber_probe.py').read())
from scipy.optimize import least_squares
P,H=load(di); L=np.r_[1,t]; Y=np.column_stack((di['instant_rate_obs'],y(di)))
F=lambda q:np.r_[1/q,-np.expm1(-t/q)]
B=np.array([F(v) for v in taus]); E=Y-(H[:,None]*th[:2])@B
u,s,v=np.linalg.svd(E,full_matrices=False); f=v[0]/v[0,0]; g=E@f/(f@f)
print('reset residual singular values',s.round(5)); print('drift temporal',f.round(5)); print('rank1 errors',np.sqrt(np.mean((E-g[:,None]*f)**2,axis=0)).round(5));print('drift rates',g.round(4))
for tag,mat in [('exp', lambda q:[q*F(q)]),('exp+linear',lambda q:[q*F(q),L]),('two branch+linear',lambda q:[*B,L]),('exp+two branch+linear',lambda q:[q*F(q),*B,L])]:
 def fit(q):
  V=np.array(mat(float(q[0]))); return V.T@np.linalg.lstsq(V.T,f,rcond=None)[0]
 q=least_squares(lambda q:fit(q)-f,[1],bounds=(.05,25)).x
 print(tag,q,np.round(fit(q)-f,5))
variants={'train+diag':m._features(tr,di)[n:],'diag':m._features(di,di)[:len(H)],'deploy scalers':None,'deploy+diag':m._features(te,di)[len(te['lab_id']):]}
raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw'],di['coupon_features_raw']]); ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); lab=np.r_[tr['lab_id'],te['lab_id'],di['lab_id']]; lab0=lab[:len(ref)]; strata=np.r_[tr['stratum'],te['stratum']];
for j in np.unique(lab):
 a=ref[lab0==j]; raw[lab==j]=(raw[lab==j]-np.median(a.mean(0)))/np.median(a.std(0))
_,i,c=np.unique(np.c_[lab0,strata],axis=0,return_inverse=True,return_counts=True); w=1/c[i,None]; fill=np.nansum(raw[:len(ref)]*w,0)/np.sum(w*~np.isnan(raw[:len(ref)]),0); variants['deploy scalers']=np.where(np.isnan(raw[len(ref):]),fill,raw[len(ref):]);
Xraw=np.asarray(di['coupon_features_raw']); complete=np.isfinite(Xraw).all(1); print('raw feature correlations and complete fits by lab')
for j in np.unique(di['lab_id']):
 a=(di['lab_id']==j)&complete; print(j,'n',sum(a),'g',g[a].round(4),'theta offset residual',(g[a]-np.nansum(Xraw[a]*th[3:],axis=1)).round(4))
for tag,X in variants.items():
 D=th[2]+X@th[3:]; p=np.c_[np.ones(len(H)),D]; b=np.linalg.lstsq(p,g,rcond=None)[0]; rr=g-p@b; print(tag,'affine drift',b.round(4),'rmse',np.sqrt((rr**2).mean()).round(5),'by lab',[(j,np.round(rr[di['lab_id']==j],3).tolist()) for j in np.unique(di['lab_id'])]);
print('upstream',__import__('_verified_upstream').__file__)
