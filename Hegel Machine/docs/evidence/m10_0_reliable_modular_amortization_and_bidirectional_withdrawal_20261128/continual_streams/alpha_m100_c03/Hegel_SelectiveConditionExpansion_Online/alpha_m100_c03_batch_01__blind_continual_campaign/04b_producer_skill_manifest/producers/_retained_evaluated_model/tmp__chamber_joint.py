exec(open('/tmp/chamber_basis.py').read().split('for label,y')[0])
np.set_printoptions(precision=5,suppress=True,linewidth=130)
labs=dg['lab_id']; n=len(labs); U=m['_features'](dg,dg)[:n]; D=w[2]+U@w[3:]; J=np.isnan(dg['coupon_features_raw']).any(1); one=np.ones(n); z=hold[:,None]-C[:,0,:2]/(-np.expm1(-t[0]/tau)); pastnorm=past/np.std(past,axis=0)
for name,P in [('compact',np.c_[one,hold,D]),('states',np.c_[one,hold,D,z]),('separate features',np.c_[one,hold,U]),('drift plus missing',np.c_[one,hold,D,np.isnan(dg['coupon_features_raw']).astype(float)])]:
 for mode in ['fixed','free','free no slope']:
  def get(ix):
   def prof(s):
    A=np.c_[f(s[0]),f(s[1]),L] if mode!='free no slope' else np.c_[f(s[0]),f(s[1])]; G=np.linalg.lstsq(P[ix],Y[ix],rcond=None)[0]@np.linalg.pinv(A.T); return A,G
   def fun(s):
    A,G=prof(s); return (P[ix]@G@A.T-Y[ix]).ravel()
   s=tau if mode=='fixed' else least_squares(fun,tau,bounds=([.5,2.5],[.85,6.])).x
   A,G=prof(s); return P@G@A.T-Y,s,G
  e,s,g=get(np.ones(n,bool)); er=np.zeros_like(Y)
  for lab in np.unique(labs):
   ix=labs==lab; er[ix]=get(~ix)[0][ix]
  print(name,mode,'times',s,'full',np.mean(e**2)**.5,'worst lab',np.max([[np.mean(er[labs==l,:-1]**2)**.5,np.mean(er[labs==l,-1]**2)**.5] for l in np.unique(labs)],axis=0),'present/missing',*[np.sqrt(np.mean(er[j]**2,axis=0))[[-2,-1]] for j in (~J,J)]); print(g.T) if name=='compact' else None
print('missing residual detail'); print(np.c_[np.flatnonzero(J),labs[J],dg['stratum'][J],dg['coupon_features_raw'][J],U[J]])
