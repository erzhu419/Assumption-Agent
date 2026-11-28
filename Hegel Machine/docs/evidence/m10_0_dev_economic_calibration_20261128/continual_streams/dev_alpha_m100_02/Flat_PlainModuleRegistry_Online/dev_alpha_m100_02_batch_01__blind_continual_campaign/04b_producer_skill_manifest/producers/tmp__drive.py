exec(open('/tmp/chamber.py').read().split("print('chamber drift")[0])
np.set_printoptions(precision=5,suppress=True,linewidth=170)
for label, x in [('train+diagnostic',xc1),('all',xc2),('frozen',xc3)]:
 r=np.c_[np.ones(len(x)),x]@w[2:]; e=A[:,:2]-hold[:,None]*w[:2]; print(label,'amplitude correction fit',np.linalg.lstsq(np.c_[np.ones(len(r)),r],e,rcond=None)[0], 'drift fit',np.linalg.lstsq(np.c_[np.ones(len(r)),r],A[:,2],rcond=None)[0]); print('r, amp correction, drift, missing');print(np.c_[r,e,A[:,2],np.isnan(c['coupon_features_raw']).sum(1)])
for p in (tr,te):
 a=p['aux_chamber_telemetry']; print('telemetry min max',np.min(a,0),np.max(a,0)); print('telemetry rows',a[:8]); print('aux correlation',np.corrcoef(np.c_[a,np.nan_to_num(p['coupon_features_raw']),p['hold_channel_raw']].T)[:4,4:]);
