from rheology_cv import *
refs=[panel([train]),panel([pilot])]
atr,ft=prep(train,refs[0]); ap,fp=prep(pilot,refs[1]); zt=fit([(atr,np.r_[.012,np.full(len(t)-1,.027)])]); zp=fit([(ap,np.r_[.0025,np.full(len(t)-1,.005)])]); print('train fit',zt,'pilot fit',zp,'fills',ft,fp)
for name,data,ref in [('train',train,refs[0]),('pilot',pilot,refs[1])]:
 m=np.any(~np.isfinite(data['coupon_features_raw']),axis=1)
 for label,fill in [('own',None),('train',ft),('pilot',fp),('zero',np.zeros(4))]:
  a,_=prep(data,ref,fill); e=err(a,zp if name=='pilot' else zt); print(name,label,'missing n',m.sum(),'missing coupon medians',np.median(e[m,1:],axis=1).round(4),'complete MAE med',np.median(abs(e[~m,1:])))
 print('rate,curve residual MAD',np.median(abs(e),axis=0).round(4))
