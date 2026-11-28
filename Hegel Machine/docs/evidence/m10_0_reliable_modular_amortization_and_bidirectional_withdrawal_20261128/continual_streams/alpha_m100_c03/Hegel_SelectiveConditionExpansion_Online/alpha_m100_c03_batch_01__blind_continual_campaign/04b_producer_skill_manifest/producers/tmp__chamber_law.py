exec(open('/tmp/chamber_basis.py').read().split('for label,y')[0])
np.set_printoptions(precision=5,suppress=True,linewidth=130)
labs=dg['lab_id']; n=len(labs); U=m['_features'](dg,dg)[:n]; D=w[2]+U@w[3:]; A=np.c_[f(tau[0]),f(tau[1]),L]; z=hold[:,None]-C[:,0,:2]/(-np.expm1(-t[0]/tau)); B=hold[:,None]*w[:2]@A[:,:2].T
for name,P,base in [('free_hold',np.c_[np.ones(n),hold,D],0*B),('fixed_hold',np.c_[np.ones(n),D],B),('fixed_hold+states',np.c_[np.ones(n),D,z],B),('fixed_hold+drift^2',np.c_[np.ones(n),D,D*D],B),('fixed_hold+product',np.c_[np.ones(n),D,D*hold],B),('fixed_hold+free_features',np.c_[np.ones(n),U],B)]:
 def coef(ix):
  return np.linalg.lstsq(P[ix],Y[ix]-base[ix],rcond=None)[0]@np.linalg.pinv(A.T)
 G=coef(slice(None)); er=np.zeros_like(Y)
 for l in np.unique(labs):
  ix=labs==l; er[ix]=base[ix]+P[ix]@coef(~ix)@A.T-Y[ix]
 print(name,'coeff',G.T,'worst lab strain/rate',np.max([np.r_[np.mean(er[labs==l,:-1]**2)**.5,np.mean(er[labs==l,-1]**2)**.5] for l in np.unique(labs)],axis=0),'worst absolute',np.max(abs(er),axis=0)[[-2,-1]])
 if name=='fixed_hold':
  J=Y-base-P@G@A.T; print('residual sv',np.linalg.svd(J,compute_uv=False)[:5]); print('residual last and rate vs drift',np.c_[D,J[:,-2:]])
