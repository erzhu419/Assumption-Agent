exec(open('/tmp/law98.py').read().split('for mode in range')[0])
from scipy.optimize import least_squares
x=m['_standardize'](a,a)[0]; q=b[2]+x@b[3:]; keep=np.isfinite(a['coupon_features_raw']).all(1); lab=a['lab_id']; G=h[:,None]*(b[:2]@T); D=Y-G
H=q[keep]@D[keep]/(q[keep]@q[keep]); print('t',t); print('drift response',H[:-1],H[-1]/.7); print('q and actual initial drift',np.c_[q,D[:,-1]/.7]); print('residual temporal RMS',np.sqrt(np.mean((D[keep]-q[keep,None]*H)**2,axis=0)))
models={
 'exp':([1.8],lambda z:z[0]*(-np.expm1(-t/z[0]))),
 'rational':([2.],lambda z:t/(1+t/z[0])),
 'rational2':([2.],lambda z:t/(1+(t/z[0])**2)),
 'tanh':([1.8],lambda z:z[0]*np.tanh(t/z[0])),
 'log':([1.],lambda z:z[0]*np.log1p(t/z[0])),
 'clip':([1.8],lambda z:np.minimum(t,z[0])),
 'exp taper':([4.],lambda z:t*np.exp(-t/z[0])),
 'exp and linear':([1.8,.07],lambda z:(1-z[1])*z[0]*(-np.expm1(-t/z[0]))+z[1]*t),
 'exp and quad':([1.8,.07],lambda z:z[0]*(-np.expm1(-t/z[0]))+z[1]*t*t),
 'sqrt saturation':([1.8],lambda z:t/np.sqrt(1+(t/z[0])**2)),
 'rational power':([2.,1.],lambda z:t/(1+t/z[0])**z[1]),
 'exp gain':([2.,.9],lambda z:z[1]*z[0]*(-np.expm1(-t/z[0]))),
 'exp quotient':([2.,1.],lambda z:t/(1+(t/z[0])**z[1])),
}
for name,(seed,fun) in models.items():
 def model(idx):
  op=least_squares(lambda z:(q[idx,None]*fun(z)-D[idx,:-1]).ravel(),seed,bounds=([.03]+[-3]*(len(seed)-1),[40]+[8]*(len(seed)-1)))
  E=q[:,None]*np.r_[fun(op.x),.7]-D;return op.x,E
 p,E=model(keep); cv=np.array([model(keep&(lab!=g))[1][i] for i,g in enumerate(lab)]); print(name,p, 'fit curve',np.sqrt(np.mean(E[keep,:-1]**2)), 'worst lab curve',max(np.sqrt(np.mean(cv[keep&(lab==g),:-1]**2)) for g in np.unique(lab)), 'worst lab rate',max(np.sqrt(np.mean(cv[keep&(lab==g),-1]**2))/.7 for g in np.unique(lab)))
