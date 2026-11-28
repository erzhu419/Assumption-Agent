exec(open('/tmp/chamber_basis.py').read().split('for label,y')[0])
np.set_printoptions(precision=5,suppress=True,linewidth=150)
A=np.column_stack([f(tau[0]),f(tau[1]),L]); _,co=fit(A.T,Y)
z=hold[:,None]-C[:,0,:2]/(-np.expm1(-t[0]/tau)); gap=hold[:,None]-z
print('tau',tau,'theta',w,'chamber amplitudes, slope, hold, z, strain0'); print(np.c_[co,hold,z,dg['strain_0']]); print('response coeff correlations',np.corrcoef(np.c_[co,hold,z,past,dg['strain_0'],X].T)[:3,3:]); print('estimated slopes and baseline drift',np.c_[co[:,2],R[:,2:]@w[2:]])
for name,P in [('hold/z',np.c_[hold,z]),('gap',gap),('hold/z/rawpast',np.c_[hold,past]),('hold/z/strain0',np.c_[hold,z,dg['strain_0']]),('hold/z/features',np.c_[hold,z,X]),('hold/z/products',np.c_[hold,z,hold[:,None]*z,z*z,z.prod(1),hold**2]),('hold/z/past squares',np.c_[hold,z,past**2]),('hold/z/last',np.c_[hold,z,past[:,-1]]),('hold/z/zsquare',np.c_[hold,z,z*z]),('hold/z/drift',np.c_[hold,z,R[:,2:]@w[2:]])]:
 P=np.c_[np.ones(len(P)),P]; coef=np.linalg.lstsq(P,co,rcond=None)[0]; pred=np.zeros_like(Y)
 for lab in np.unique(dg['lab_id']):
  ix=dg['lab_id']==lab; pred[ix]=P[ix]@np.linalg.lstsq(P[~ix],co[~ix],rcond=None)[0]@A.T
 err=pred-Y; full=P@coef@A.T-Y; print(name,'full',np.sqrt(np.mean(full**2,axis=0))[[0,5,-2,-1]],'outer lab',[(int(l),np.round(np.sqrt(np.mean(err[dg['lab_id']==l]**2,axis=0))[[0,5,-2,-1]],4)) for l in np.unique(dg['lab_id'])]); print('  coeff',coef.T)
