from chamber_cv import *
np.set_printoptions(precision=5,suppress=True)
clock=tau*np.array([1.,1.18]); _,A=prof(np.log(clock)); print('clocks',clock,'keys',list(tr),list(te),list(q)); print('times',t,dt)
for k,D in [('tr',tr),('te',te),('q',q)]:
 print(k)
 for l in np.unique(D['lab_id']):
  R=D['ref_coupons'][D['lab_id']==l]; print(l,np.round(R.mean(0),3),np.round(R.std(0),3))
for fm in ['self','joint','frozen']:
 z=feat(fm); dr=beta[2]+z@beta[3:]; miss=np.isnan(q['coupon_features_raw']).any(1)
 print(fm, 'complete',~miss)
 for j in range(3):
  print('mode',j)
  for name,B in [('hold',np.c_[hq]),('linear',np.c_[hq,dr]),('bilinear',np.c_[hq,dr,hq*dr]),('quadratic',np.c_[hq,dr,hq*dr,hq**2,dr**2]),('history',np.c_[hq,dr,prior._gap(tau,pq,hq,dt)]),('raw',np.c_[hq,z]),('rawlab',np.c_[hq,z,(q['lab_id'][:,None]==np.arange(1,4)).astype(float)])]:
   B=np.c_[np.ones(len(B)),B]; b=np.linalg.lstsq(B[~miss],A[~miss,j],rcond=None)[0]; r=A[:,j]-B@b; print(name,'rms',np.sqrt(np.mean(r[~miss]**2)),'missing',r[miss], 'coef',b)
  if fm=='self':
   B=np.c_[np.ones(len(hq)),hq,dr]; r=A[:,j]-B@np.linalg.lstsq(B[~miss],A[~miss,j],rcond=None)[0]; print('by lab',[(int(l),np.round(r[q['lab_id']==l],4).tolist()) for l in np.unique(q['lab_id'])])
