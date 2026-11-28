exec(open('/tmp/probe.py').read().split('for name,u,x in')[0])
p,h=_loads(ca,q); y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; Y=np.c_[y,r*.35]; lab=q['lab_id']; st=q['stratum']; B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],.35*np.r_[1/tau,1]])
keys=('lab_id','stratum','coupon_features_raw','ref_coupons'); trte={k:np.concatenate([tr[k],te[k]]) for k in keys}
for label,a in [('train',tr),('test',te),('diag',q)]:
 print(label,'missing',np.isnan(a['coupon_features_raw']).sum(axis=0),'labs',[(int(l),round(np.median(a['ref_coupons'][a['lab_id']==l].mean(0)),3),round(np.median(a['ref_coupons'][a['lab_id']==l].std(0)),3)) for l in np.unique(a['lab_id'])]); print('raw feature ranges',np.nanmin(a['coupon_features_raw'],axis=0),np.nanmax(a['coupon_features_raw'],axis=0))
def transform(ref, target):
 x=target['coupon_features_raw'].copy(); X=ref['coupon_features_raw'].copy()
 for l in np.unique(target['lab_id']):
  z=ref['lab_id']==l; panel=ref['ref_coupons'][z]; c=np.median(panel.mean(0)); s=np.median(panel.std(0)); x[target['lab_id']==l]=(x[target['lab_id']==l]-c)/s; X[z]=(X[z]-c)/s
 _,i,n=np.unique(np.c_[ref['lab_id'],ref['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[i,None]; m=np.nansum(w*X,axis=0)/np.sum(w*np.isfinite(X),axis=0); return np.where(np.isnan(x),m,x)
Z=np.zeros((len(h),2));
for j,w in enumerate(dt): Z=np.exp(-w/tau)*Z-np.expm1(-w/tau)*p[:,j,None]
for alignment,X in [('joint3',xq[:,1:]),('train_test',transform(trte,q)),('diagnostic',transform(q,q)),('raw',np.nan_to_num(q['coupon_features_raw']))]:
 D=th[2]+X@th[3:]; g=np.ones((len(h),1)); h1=h[:,None]; d1=D[:,None]; print('\n',alignment,'D range',D.min(),D.max());
 families={'H D':np.c_[g,h1,d1], 'H X':np.c_[g,h1,X], 'H Z D':np.c_[g,h1,Z,d1], 'H Z X':np.c_[g,h1,Z,X], 'H D D²':np.c_[g,h1,d1,d1**2], 'H D HD':np.c_[g,h1,d1,h1*d1], 'H D D² HD':np.c_[g,h1,d1,d1**2,h1*d1], 'H D H²':np.c_[g,h1,d1,h1**2], 'H D arctan':np.c_[g,h1,np.arctan(d1)], 'H D tanh':np.c_[g,h1,np.tanh(d1)], 'H D signed-sqrt':np.c_[g,h1,np.sign(d1)*np.sqrt(abs(d1))], 'H D H*D²':np.c_[g,h1,d1,h1*d1,h1*d1**2], 'H D D³':np.c_[g,h1,d1,d1**3], 'H D H/(1+D²)':np.c_[g,h1,d1,h1/(1+d1**2)], 'H D |D|':np.c_[g,h1,d1,abs(d1)], 'H X HX':np.c_[g,h1,X,h1*X]}
 for name,F in families.items():
  E=np.empty_like(Y); emax=[]
  for l in np.unique(lab):
   m=lab==l; E[m]=F[m]@np.linalg.lstsq(F[~m],Y[~m],rcond=None)[0]-Y[m]; emax.append([np.mean(E[m,:-1]**2)**.5,np.mean(E[m,-1]**2)**.5/.35]);
  rms=[np.mean(E[:,:-1]**2)**.5,np.mean(E[:,-1]**2)**.5/.35]; w=np.linalg.lstsq(F,np.linalg.lstsq(B,Y.T,rcond=None)[0].T,rcond=None)[0]; print(name,'LOLO',np.round(rms,5),'worst',np.round(np.max(emax,0),5), 'W' if name in ('H D','H X','H D HD') else '',np.round(w,4) if name in ('H D','H X','H D HD') else '')
