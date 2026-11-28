exec(open('/tmp/chamber_basis.py').read().split('for label,y')[0])
np.set_printoptions(precision=6,suppress=True)
labs=dg['lab_id']; U=m['_features'](dg,dg)[:len(labs)]; D=w[2]+U@w[3:]; M=np.isnan(dg['coupon_features_raw']).astype(float); j=M.any(1); z=hold[:,None]-C[:,0,:2]/(-np.expm1(-t[0]/tau)); one=np.ones(len(hold))
for kind in ('two','two+states','three','three+states'):
 for free in (False,True):
  F=np.c_[hold,D,M]; F=np.c_[F,z] if 'states' in kind else F
  def model(i):
   def profile(s):
    A=np.c_[f(s[0]),f(s[1])]; A=np.c_[A,L] if 'three' in kind else A
    G=np.linalg.lstsq(F[i],Y[i],rcond=None)[0]@np.linalg.pinv(A.T)
    return F@G@A.T-Y,G
   s=least_squares(lambda s:profile(s)[0][i].ravel(),tau,bounds=([.5,2.5],[.85,6.])).x if free else tau
   return (*profile(s),s)
  e,g,s=model(np.ones(len(labs),bool)); e0,_,_=model(~j); err=np.zeros_like(Y)
  for lab in np.unique(labs):
   i=labs!=lab; err[~i]=model(i)[0][~i]
  print(kind,free,s,'coef hold/drift/missing/(states):',g.T,'fit',np.mean(e**2)**.5,'complete fit',np.mean(e0[~j]**2)**.5,'worst lab',[max(np.mean(err[labs==l,k]**2)**.5 for l in np.unique(labs)) for k in (slice(None,-1),-1)],'worst missing time',np.sqrt(np.mean(err[j]**2,axis=0))[[-2,-1]])
print('diagnostic fill',np.unique(np.where(M,U,np.nan),axis=0)); print('deployment fill',np.nanmean(np.where(np.isnan(te['coupon_features_raw']),x[len(tr['lab_id']):],np.nan),axis=0)); print('train rate/curve scale',np.quantile(abs((tr['strain_curves_obs']-tr['strain_0'][:,None])-m['_design'](tau,x[:len(tr['lab_id'])],*m['_loads'](tr,b),dt,t)[0]@w),[.25,.5,.75,.9,.95,.99]))
