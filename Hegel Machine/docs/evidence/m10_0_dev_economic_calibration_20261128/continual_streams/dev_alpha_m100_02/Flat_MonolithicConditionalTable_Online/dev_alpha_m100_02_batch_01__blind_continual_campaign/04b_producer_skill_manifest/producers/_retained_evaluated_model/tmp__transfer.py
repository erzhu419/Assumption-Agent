exec(open('/tmp/probe.py').read().split('for g,ss in')[0])
p,h,f=sd
D=np.vstack((np.r_[1/tau,1],np.c_[-np.expm1(-t[:,None]/tau),t])); k=np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0.][:,None]/tau)*(-np.expm1(-dt[:,None]/tau)); z=p@k
V=Y(di)-h[:,None]*(D[:,:2]@th[:2]); q0=f@th[2:];
fg=m._features(tr,{k:np.concatenate((te[k],di[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')})[1][-len(h):]; fd=m._features(di,di)[0];
for tag,q in [('main',q0),('all',th[2]+fg@th[3:]),('diag',th[2]+fd@th[3:])]:
 print('\nFEATURE',tag,q)
 for drift in [q[:,None],np.c_[q,q**2],np.c_[q,np.tanh(q)],np.c_[q,np.tanh(2*q)],np.c_[q,q**3],np.c_[q,q**2,q**3]]:
  n=len(h); nc=V.shape[1]; models=[]
  for mem in [[],[z[:,0,None]*D[:,0],z[:,1,None]*D[:,1]]]:
   A=np.stack(mem+[v[:,None]*np.eye(nc)[j] for v in drift.T for j in range(nc)],axis=2); rows=np.arange(n); E=np.zeros_like(V)
   for i in rows:
    X=A[rows!=i].reshape(-1,A.shape[2]); b=np.linalg.lstsq(X,V[rows!=i].ravel(),rcond=None)[0]; E[i]=V[i]-A[i]@b
   b=np.linalg.lstsq(A.reshape(-1,A.shape[2]),V.ravel(),rcond=None)[0]; models.append([np.round(np.sqrt((E**2).mean(0))[[0,1,-1]],4),np.round(b[:len(mem)],3)])
  print(drift.shape[1],models)
 print('free q profile',np.linalg.lstsq(np.c_[z,q],V,rcond=None)[0])
print('missing diagnostic:',np.where(np.isnan(di['coupon_features_raw'])))
