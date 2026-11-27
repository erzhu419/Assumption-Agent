exec(open('/tmp/rheo_probe.py').read().split("print('train RMSE'")[0])
P,H,D=c; n=len(H)
X=m._features(dg,dg)[-n:]; R=w[2]+X@w[3:]; z=dg['coupon_features_raw']; miss=np.isnan(z).any(axis=1)
B=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); F=(H[:,None]*w[:2])@B.T; Y=yd-F
for tag,R0 in [('balanced',R),('zero-missing',R-np.where(np.isnan(z),X*w[3:],0).sum(axis=1))]:
 for v in [~miss,np.ones(n,dtype=bool)]:
  coeff=np.linalg.lstsq((R0[v,None,None]*B).reshape(-1,2),Y[v].ravel(),rcond=None)[0]
  e=R0[:,None]*(B@coeff)-Y
  print(tag, 'fit_n',sum(v),'gain',coeff.round(5),'complete/missing rmse',[(np.sqrt(np.mean(e[s,:-1]**2)),np.sqrt(np.mean(e[s,-1]**2))) for s in [~miss,miss]])
  if tag=='balanced' and not np.all(v):
   basis=B@coeff; dr=Y@basis/(basis@basis)-R0
   print('missing rows, estimated drift correction, implied missing value',[(int(i), dr[i], X[i,np.isnan(z[i])] + dr[i]/w[3:][np.isnan(z[i])]) for i in np.flatnonzero(miss)])
   print('complete row residuals', np.c_[dg['lab_id'][~miss],dr[~miss]].round(5))
