import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m98_chamber import *
keys=('lab_id','stratum','coupon_features_raw','ref_coupons')
empty={k:tr[k][:0] for k in keys}
def report(label,tx,dx):
 sp=m._specimen(tr,tx,bridge)
 tau,th=m._fit(sp,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs'])
 print(label,'tau',np.round(tau,5),'theta',np.round(th,5))
 for title,g,x,oth in [('train',tr,tx,None),('diag',diag,dx,'sat')]:
  sp=m._specimen(g,x,bridge); b,rr=m._branches(sp,dt,t,tau); drift=sp[2]@th[2:]; ec=b@th[:2]+drift[:,None]*(t if oth is None else -np.expm1(-t))-(g['strain_curves_obs']-g['strain_0'][:,None]); er=rr@th[:2]+drift-g['instant_rate_obs']; print(title,'curve med & RMSE',np.round([np.median(abs(ec)),np.sqrt(np.mean(ec**2))],6),'rate RMSE',round(np.sqrt(np.mean(er**2)),6),'by lab',[(int(j),round(np.sqrt(np.mean(ec[g['lab_id']==j]**2)),6),round(np.sqrt(np.mean(er[g['lab_id']==j]**2)),6)) for j in np.unique(g['lab_id'])]);
  if title=='diag': print('diag rate errors',np.round(er,5))
xt,_=m._features(tr,empty); xd,_=m._features(diag,empty); xa,_=m._features(tr,te); xad,xda=m._features(tr,diag)
report('joint train+test, diag own',xa,xd)
report('train own, diag own',xt,xd)
report('joint train+diag',xad,xda)
print('test/diag keys',list(te),list(diag)); print('diagnostic missing rows',np.flatnonzero(np.isnan(diag['coupon_features_raw']).any(axis=1))); print('equal panel rows',np.sum((te['ref_coupons'][:,None,:]==diag['ref_coupons'][None,:,:]).all(axis=-1)))
