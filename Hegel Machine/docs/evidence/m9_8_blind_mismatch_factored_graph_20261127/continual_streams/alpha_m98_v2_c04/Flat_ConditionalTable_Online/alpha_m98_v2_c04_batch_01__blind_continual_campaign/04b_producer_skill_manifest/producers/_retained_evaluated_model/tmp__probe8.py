exec(open('/tmp/joint.py').read().split("for kind in ('standard'")[0])
from scipy.optimize import least_squares
cell=np.c_[labs,di['stratum']]; _,ii,counts=np.unique(cell,axis=0,return_inverse=True,return_counts=True); w=1/counts[ii]; z=di['coupon_features_raw'].copy()
for lab in np.unique(labs):
    q=di['ref_coupons'][labs==lab]; z[labs==lab]=(z[labs==lab]-np.median(q.mean(0)))/np.median(q.std(0))
mask=np.isnan(z); print('missing rows',np.where(mask)); fixed=np.log([.472,2.47]); B=basis(fixed); amp=np.linalg.lstsq(B,y.T,rcond=None)[0].T
for tag,fill in [('joint',None),('zero',np.zeros(4)),('local-balanced',np.nansum(w[:,None]*z,0)/np.sum(w[:,None]*~mask,0)),('local-unweighted',np.nanmean(z,0))]:
 zz=prep('diag') if fill is None else np.where(mask,fill,z); r=th[2]+zz@th[3:]; X=np.c_[np.ones(N),hd,r]; pred,c,u=run([X]*3,[]); e=pred-y; oo=np.empty_like(y)
 for lab in np.unique(labs):
  ix=np.flatnonzero(labs==lab); oo[ix]=(run([X]*3,ix)[0]-y)[ix]
 print(tag,'fill',fill,'coef',c.reshape(3,3).round(5),'taus',u,'cv',[(l,np.sqrt(np.mean(oo[labs==l,:-1]**2)),np.sqrt(np.mean(oo[labs==l,-1]**2))) for l in np.unique(labs)]); print('rows [lab missing drift strain-last rate]',np.c_[labs,mask.any(1),r,e[:,-2:]].round(5))
 if tag=='local-balanced':
  clean=~mask.any(1); pred0,c0,u0=run([X]*3,np.flatnonzero(~clean)); print('complete-only coefficients',c0.reshape(3,3),'taus',u0,'errors',np.sqrt(np.mean((pred0[clean]-y[clean])**2,0))); print('implied drift with rounded law',np.c_[r, (amp[:,0]-.62*hd)/.166,(amp[:,1]-.28*hd)/1.38,(amp[:,2]-.03*hd)/.1].round(4))
