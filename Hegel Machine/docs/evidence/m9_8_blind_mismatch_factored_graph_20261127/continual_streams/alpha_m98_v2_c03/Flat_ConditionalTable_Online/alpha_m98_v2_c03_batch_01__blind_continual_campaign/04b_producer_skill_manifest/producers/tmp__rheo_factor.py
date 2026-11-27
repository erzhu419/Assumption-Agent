exec(open('/tmp/rheo_probe.py').read().split("print('train RMSE'")[0])
P,H,D=c; n=len(H); R=D@w[2:];
B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]])
A=np.linalg.lstsq(B,yd.T,rcond=None)[0].T; A2=np.linalg.lstsq(B[:,:2],yd.T,rcond=None)[0].T
base=m._design(c,dt,t,tau)[:,-1,:2]*tau*w[:2]
cal=d['calibration_load_cells']; loads=cal['true_channel_loads']; raw=cal['bridge_raw']; fit=np.linalg.lstsq(np.c_[np.ones(len(loads)),loads],raw,rcond=None)[0]; inv=np.linalg.inv(fit[1:]); hp=(dg['hold_channel_raw']-fit[0])@inv; pp=(dg['past_channel_raw']-fit[0])@inv
print('columns lab,stratum, hold, drift, amps, amp/weight-hold, driftobs, channels')
print(np.c_[dg['lab_id'],dg['stratum'],H,R,A2,A2/w[:2]-H[:,None],A[:,-1],hp].round(4)); print('no drift rmse',np.sqrt(np.mean((A2@B[:,:2].T-yd)**2,0)).round(5))
for name,Z in [('hold',H[:,None]),('baseline',base),('drift',R[:,None]),('hold+drift',np.c_[H,R]),('hold+drift2',np.c_[H,R,R**2]),('hold+drift+interaction',np.c_[H,R,H*R]),('hold+features',np.c_[H,D[:,1:]]),('hold+baseline+drift',np.c_[H,base,R]),('holdchannels+drift',np.c_[hp,R]),('hold+drift+past',np.c_[H,R,P]),('hold+drift+lab',np.c_[H,R,np.eye(4)[dg['lab_id']][:,1:]]),('hold+drift+stratum',np.c_[H,R,np.eye(max(dg['stratum'])+1)[dg['stratum']][:,1:]])]:
 Z=np.c_[np.ones(n),Z]; co=np.linalg.lstsq(Z,A2,rcond=None)[0]; pred=np.zeros_like(A2)
 for g in np.unique(dg['lab_id']):
  v=dg['lab_id']==g; pred[v]=Z[v]@np.linalg.lstsq(Z[~v],A2[~v],rcond=None)[0]
 print(name,'fit',np.sqrt(np.mean((Z@co-A2)**2,0)).round(5),'worst lab amp',np.max([np.sqrt(np.mean((pred-A2)[dg['lab_id']==g]**2,0)) for g in np.unique(dg['lab_id'])],0).round(5),'coef',co.T.round(4))
print('diag features raw'); print(dg['coupon_features_raw'].round(3)); print('archive trace excerpt'); import json; j=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))[-1]['facility_telemetry_trace']; print(str(j)[:420])
