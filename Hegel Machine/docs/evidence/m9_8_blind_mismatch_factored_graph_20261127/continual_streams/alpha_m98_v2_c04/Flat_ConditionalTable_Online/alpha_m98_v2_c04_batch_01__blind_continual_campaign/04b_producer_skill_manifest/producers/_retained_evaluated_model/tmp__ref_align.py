exec(open('/tmp/baseline.py').read().split('for name,b,xx')[0])
np.set_printoptions(precision=5,suppress=True,linewidth=140)
for name,b in [('train',tr),('test',te),('diag',di)]:
 print(name)
 for lab in np.unique(b['lab_id']):
  rows=b['lab_id']==lab; q=b['ref_coupons'][rows]; xx=b['coupon_features_raw'][rows]; print(lab,len(q),[np.median(q.mean(0)),np.median(q.std(0))], 'missing',np.isnan(xx).sum(0),'means',np.nanmean(xx,0))
 if 'aux_chamber_telemetry' in b:
  a=b['aux_chamber_telemetry']; print('telemetry first',a[:8],'range',np.nanpercentile(a,[0,50,100],axis=0))
print('eval',t,'dt',dt)
