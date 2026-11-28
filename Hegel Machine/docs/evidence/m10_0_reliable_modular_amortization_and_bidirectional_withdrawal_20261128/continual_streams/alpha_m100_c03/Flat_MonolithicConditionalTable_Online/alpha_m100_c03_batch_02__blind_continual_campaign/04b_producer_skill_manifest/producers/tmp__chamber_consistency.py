from chamber_probe import *
B=A[:,:,:2]@w[:2]; br=R[:,:2]@w[:2]; G=r-br
Q=-np.expm1(-ts)
print('unit buildup prediction with independently observed initial rate: median/maxabs',np.median(abs(B+G[:,None]*Q-y)),np.max(abs(B+G[:,None]*Q-y)))
q=(y-B)@Q/(Q@Q); print('curve inferred drift vs rate inferred drift RMSE',np.mean((q-G)**2)**.5)
print('observed/inferred drift range',np.min(q),np.max(q)); print('feature construction options, drift RMSE and split by missing values')
missing=np.isnan(di['coupon_features_raw']).any(axis=1)
for label,x0 in [('train+diag',x),('diag only',m._features(di,di)[:len(h)]),('test+diag',m._features(te,di)[-len(h):])]:
 g0=w[2]+x0@w[3:]; print(label,[np.mean((q[mask]-g0[mask])**2)**.5 for mask in (np.ones(len(h),bool),~missing,missing)])
for label,x0 in [('train+test ref',m._features(tr,te)),('train only ref',m._features(tr,tr)[:len(tr['lab_id'])])]:
 pp,hh=m._loads(tr,b); k=len(hh); yy=tr['strain_curves_obs']-tr['strain_0'][:,None]; rr=tr['instant_rate_obs']; tt,ww=m._fit(x0[:k],pp,hh,dt,ts,yy,rr); aa,bb=m._design(tt,x0[:k],pp,hh,dt,ts); print(label,tt,ww,'median strain/rate',np.median(abs(aa@ww-yy)),np.median(abs(bb@ww-rr)))
