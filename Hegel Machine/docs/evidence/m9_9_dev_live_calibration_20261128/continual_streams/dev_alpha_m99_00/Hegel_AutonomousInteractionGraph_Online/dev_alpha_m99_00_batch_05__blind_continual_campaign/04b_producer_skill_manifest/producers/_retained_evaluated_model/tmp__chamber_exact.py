exec(open('/tmp/forcing.py').read().split('for name,r in D.items():')[0])
raw=c['coupon_features_raw']; full=np.isfinite(raw).all(1); print('complete by lab',[(j,int(sum(full&(labs==j)))) for j in np.unique(labs)])
for maskname,mask in [('all',np.ones(len(H),bool)),('complete',full)]:
 for dn,r in D.items():
  for extra,xx in [('base',[]),('offset',[c['strain_0']]),('history',[Z[:,0],Z[:,1]]),('features',list(old._features(c).T))]:
   F=np.column_stack([O,H,r]+xx)[mask]; YY=Y[mask]; LL=labs[mask]
   def predict(ix):
    C=np.linalg.pinv(F[ix])@YY[ix]
    def fit(v,ret=False):
     T=np.r_[taus[0],np.exp(v)]; B=basis(T); W=C@np.linalg.pinv(B).T; R=F@W@B.T-YY
     return (R,T,W) if ret else R[ix].ravel()
    v=least_squares(fit,[np.log(3.3)],bounds=(np.log(.55),np.log(18))).x
    return fit(v,True)
   R,T,W=predict(np.ones(len(YY),bool)); C=np.empty_like(R)
   for l in np.unique(LL): C[LL==l]=predict(LL!=l)[0][LL==l]
   print(maskname,dn,extra,'T',T.round(4),'in/out/maxlab',np.round([np.mean(R**2)**.5,np.mean(C**2)**.5,max(np.mean(C[LL==l]**2)**.5 for l in np.unique(LL))],5),'late/rate',np.sqrt(np.mean(C[:,-2:]**2,0)).round(5))
   if extra=='base': print('coef',W.round(6)); print('resid missing/complete',[(int(j), np.round(R[i,[-2,-1]],4).tolist()) for i,j in enumerate(np.where(~full[mask],1,0))])
