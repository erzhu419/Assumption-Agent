exec(open('/tmp/chamber_amplitude.py').read().split('zsets=')[0].replace('print(np.c_[C,H,G,D,U].round(4))','pass'))
_,xd=m._features(c,c)
complete=np.isfinite(c['coupon_features_raw']).all(axis=1)
obs=np.c_[Y,R]
for label, xx in [('joint train/test',x),('train only',m._features(a,a)[0])]:
 tt,bb=m._fit(xx,p,h,dt,t,y,r)
 aa,br=m._design(tt,xx,p,h,dt,t); e=np.c_[aa@bb-y, br@bb-r]
 print('train alignment',label,tt,bb,np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])))
for l in np.unique(lab):
 print('lab alignment',l,[(name, len(part['lab_id'][part['lab_id']==l]), np.median(part['ref_coupons'][part['lab_id']==l].mean(0)),np.median(part['ref_coupons'][part['lab_id']==l].std(0))) for name,part in [('train',a),('test',b),('diag',c)]])
def verify(tau,theta):
 basis=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.])).T
 dd=theta[2]+xd@theta[3:]; design=dd[:,None,None]*basis.T
 base=np.c_[H[:,None]*theta[:2],np.zeros(len(H))]@basis
 target=obs-base
 def coef(mask):return np.linalg.lstsq(design[mask].reshape(-1,3),target[mask].ravel(),rcond=None)[0]
 k=coef(complete); print('complete-case coefficients',k,'rounded',np.round(k,2)); print('fitted without missing',[(int(l),np.round(coef(complete & (lab!=l)),5)) for l in np.unique(lab)])
 for name, kw in [('fitted',k),('rounded',np.round(k,2))]:
  e=base+design@kw-obs
  for masklab in np.unique(lab):
   for label,mask in [('complete',complete),('missing',~complete)]:
    s=(lab==masklab)&mask
    if np.any(s): print(name,masklab,label,np.sqrt(np.mean(e[s,:-1]**2)),np.sqrt(np.mean(e[s,-1]**2)))
 e=np.empty_like(obs)
 for l in np.unique(lab):e[lab==l]=base[lab==l]+design[lab==l]@coef(complete & (lab!=l))-obs[lab==l]
 print('OOF worst lab complete strain/rate',np.max([np.sqrt(np.mean(e[complete & (lab==l)]**2,axis=0)) for l in np.unique(lab)],axis=0).round(5))
 # Compare observed-imputation conventions without changing the required benchtop aligner.
 for l in np.flatnonzero(~complete):
  j=np.flatnonzero(~np.isfinite(c['coupon_features_raw'][l]))[0]; mode=k@basis
  dtarget=((obs[l]-base[l])@mode)/(mode@mode)
  implied=xd[l,j]+(dtarget-dd[l])/theta[j+3]
  raw=c['coupon_features_raw']; print('missing',l,j,'implied/current/zero',implied,xd[l,j],0.,'lab and stratum',lab[l],c['stratum'][l])
verify(tau,theta)
