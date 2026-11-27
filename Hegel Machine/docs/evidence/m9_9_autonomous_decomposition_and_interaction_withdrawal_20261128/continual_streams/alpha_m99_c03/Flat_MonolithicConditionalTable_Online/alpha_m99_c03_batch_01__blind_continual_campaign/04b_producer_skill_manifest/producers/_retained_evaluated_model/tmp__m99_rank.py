from m99_inputs import *
D=np.vstack((E,er)); obs=np.c_[y,yr]; lab=g['lab_id']; strat=g['stratum']; n=len(lab); p,h,_=cg
miss=np.isnan(g['coupon_features_raw']).any(1)
B=np.linalg.lstsq(D,obs.T,rcond=None)[0].T
models={}
for label,xx in [('diag',m._features(g,g)[-n:]),('joint',m._features(tr,g)[-n:])]:
 dr=xx@th[3:]
 for name,F,free in [('drift',np.c_[np.ones(n),dr],False),('drift+quadratic',np.c_[np.ones(n),dr,dr**2],False),('drift+hold interaction',np.c_[np.ones(n),dr,h*dr],False),('drift+linear hold',np.c_[np.ones(n),dr,h],True),('drift+stress contrast',np.c_[np.ones(n),dr,h-p[:,-1]],False),('drift+memory',np.c_[np.ones(n),dr,h[:,None]-R0[:,:2]*tau],False),('features',np.c_[np.ones(n),xx],False)]:
  off=np.zeros((n,3)) if free else np.c_[h[:,None]*th[:2],np.zeros(n)]
  coef=np.linalg.lstsq(F,B-off,rcond=None)[0]; errors=[]
  for blocks in (lab,strat,np.arange(n),np.where(miss,1,0)):
   pp=np.zeros_like(B)
   for k in np.unique(blocks):
    u=blocks==k; pp[u]=off[u]+F[u]@np.linalg.lstsq(F[~u],(B-off)[~u],rcond=None)[0]
   e=pp@D.T-obs
   errors.append(np.max([[np.mean(e[blocks==k,:-1]**2)**.5,np.mean(e[blocks==k,-1]**2)**.5] for k in np.unique(blocks)],0))
  print(label,name,'lab/stratum/coupon/missing RMSE',np.round(errors,4).tolist(),'coef',coef.T.round(4).tolist())
print('per coupon implied chamber scalar'); xx=m._features(g,g)[-n:]; Q=B-np.c_[h[:,None]*th[:2],np.zeros(n)]; C=np.linalg.lstsq(np.c_[np.ones(n),xx@th[3:]],Q,rcond=None)[0]; da=(Q-C[0])/C[1]; print(np.c_[lab,strat,miss,xx@th[3:],da].round(4))
