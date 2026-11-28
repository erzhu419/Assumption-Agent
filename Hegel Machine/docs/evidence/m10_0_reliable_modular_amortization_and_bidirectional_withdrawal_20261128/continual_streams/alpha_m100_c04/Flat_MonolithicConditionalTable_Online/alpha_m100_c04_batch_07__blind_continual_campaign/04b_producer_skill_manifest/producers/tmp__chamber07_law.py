exec(open('/tmp/chamber07_affine.py').read().split("print('bridge',")[0])
h=m._stress(z['hold_channel_raw'],b); past=m._stress(z['past_channel_raw'],b); Y=np.c_[z['strain_curves_obs']-z['strain_0'][:,None],z['instant_rate_obs']]; time=np.r_[t,1.]; labs=z['lab_id']; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; mem=past@(-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau)); A=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); base=((h[:,None]-mem)*theta[:2])@A.T
blocks=[('diag',z,z),('train+diag',tr,z),('test+diag',te,z)]
for nm,p1,p2 in blocks:
 xx=m._features(p1,p2)[len(p1['lab_id']):]; drift=theta[2]+xx@theta[3:]; print(nm,'ranges',drift.min(),drift.max())
 for law in ['baseline','scalar drift','branch drift','branch drift+intercept','free linear features','quadratic drift','free gaps+branch drift']:
  R=Y-base
  terms=[drift[:,None]*time]
  if law!='scalar drift': terms=[drift[:,None]*v for v in [A[:,0],A[:,1],time]]
  if law=='branch drift+intercept': terms += [np.broadcast_to(v,Y.shape) for v in [A[:,0],A[:,1],time]]
  if law=='free linear features': terms=[q[:,None]*v for q in np.c_[np.ones(len(h)),xx].T for v in [A[:,0],A[:,1],time]]
  if law=='quadratic drift': terms += [drift[:,None]**2*v for v in [A[:,0],A[:,1],time]]
  if law=='free gaps+branch drift': terms += [(h-mem[:,j])[:,None]*A[:,j] for j in range(2)]
  D=np.stack(terms,axis=-1); oof=np.empty_like(Y); cs=[]
  for lab in np.unique(labs):
   s=labs==lab; c=np.linalg.lstsq(D[~s].reshape(-1,len(terms)),R[~s].ravel(),rcond=None)[0];oof[s]=(base[s]+D[s]@c-Y[s]) if law!='baseline' else base[s]+drift[s,None]*time-Y[s];cs.append(c)
  c=np.linalg.lstsq(D.reshape(-1,len(terms)),R.ravel(),rcond=None)[0]; print(law,'worst',max(np.sqrt(np.mean(oof[labs==j,:-1]**2,axis=0)).max() for j in np.unique(labs)),max(np.sqrt(np.mean(oof[labs==j,-1]**2)) for j in np.unique(labs)),'coef',c.round(5),'fold spread',np.ptp(cs,axis=0).round(5))
 print('ref means',[(j,round(np.median(p1['ref_coupons'][p1['lab_id']==j].mean(0)),4),round(np.median(z['ref_coupons'][labs==j].mean(0)),4)) for j in np.unique(labs)])
