exec(open('/tmp/chamber_basis.py').read().split('for label,y')[0])
np.set_printoptions(precision=5,suppress=True,linewidth=140)
N=len(hold); labs=dg['lab_id']; one=np.ones(N); P0=np.c_[hold,past]; Z=hold[:,None]-C[:,0,:2]/(-np.expm1(-t[0]/tau)); A=np.c_[f(.65),f(3.87),L]; co=np.linalg.lstsq(A,Y.T,rcond=None)[0].T
print('time/dt',t,dt,'amplitude+drift min/max',co.min(0),co.max(0)); print('leading amplitude equations raw/unweighted',np.linalg.lstsq(np.c_[one,hold,co[:,2]],co,rcond=None)[0]); print('strain0 and amplitude range',np.c_[dg['strain_0'],co][:8]);
ref=np.vstack([q['ref_coupons'] for q in (tr,te)]); lab=np.r_[tr['lab_id'],te['lab_id']]; x3=m['_features']({'coupon_features_raw':np.vstack([q['coupon_features_raw'] for q in (tr,te)]),'lab_id':lab,'stratum':np.r_[tr['stratum'],te['stratum']],'ref_coupons':ref},dg)[-N:]; xd=m['_features'](dg,dg)[:N]; xr=dg['coupon_features_raw'];
center=np.array([np.median(ref[lab==l].mean(0)) for l in labs]); scale=np.array([np.median(ref[lab==l].std(0)) for l in labs]); xf=np.where(np.isnan(xr),np.nanmean(x,axis=0),(xr-center[:,None])/scale[:,None]);
print('joint panel center/scale',np.unique(np.c_[labs,center,scale],axis=0)); print('missing rows',np.where(np.isnan(xr).any(1))[0]);
for label,U in [('train+diag',X),('train+test fixed',xf),('all three',x3),('diag',xd),('raw',np.nan_to_num(xr))]:
 drift=w[2]+U@w[3:];
 for memory,P in [('no memory',np.c_[one,hold]), ('two states',np.c_[one,hold,Z]),('all history',np.c_[one,P0])]:
  for name,Xp in [('drift',np.c_[P,drift]),('feat',np.c_[P,U])]:
   def fitpred(ix, jx, free):
    def profile(s):
     T=np.c_[f(s[0]),f(s[1]),L]; B=np.linalg.lstsq(Xp[ix],Y[ix],rcond=None)[0]; G=np.linalg.lstsq(T,B.T,rcond=None)[0].T; return G,T
    if free:
     def err(s):
      G,T=profile(s); return (Xp[ix]@G@T.T-Y[ix]).ravel()
     s=least_squares(err,[.65,3.88],bounds=([.5,2],[.9,5.5])).x
    else: s=tau
    G,T=profile(s); return Xp[jx]@G@T.T,G,s
   for free in (False,True) if label in ('train+diag','train+test fixed') and name=='drift' else (False,):
    err=np.zeros_like(Y)
    for l in np.unique(labs):
     ix=labs!=l; err[~ix]=fitpred(ix,~ix,free)[0]-Y[~ix]
    pred,G,s=fitpred(np.ones(N,bool),np.ones(N,bool),free)
    print(label,memory,name,free,'fit',round(np.mean((pred-Y)**2)**.5,5),'worst lab strain/rate',np.max([np.r_[np.mean(err[labs==l,:-1]**2)**.5,np.mean(err[labs==l,-1]**2)**.5] for l in np.unique(labs)],axis=0),'times',s)
    if label=='train+test fixed' and name=='drift': print('  coefficients',G.T)
