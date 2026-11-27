exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
for name,g,z in [('train',tr,x[0]),('pilot',pi,x[2])]:
 tau,w,sc=s._fit(g,z,b,dt,t); y=s._obs(g); r=y-s._design(g,z,b,dt,t,tau)@w
 slope=np.median(r[:,:-1]/t,axis=1); nonlinear=np.median(abs(r[:,:-1]-slope[:,None]*t),axis=1)
 print(name,'mad curve/rate',np.median(abs(r),axis=0).round(4),'rate minus slope',np.quantile(abs(r[:,-1]-slope),[.5,.9,1]).round(4))
 for j in (np.arange(len(r)) if name=='pilot' else np.argsort(abs(slope))[-12:]):
  print(j,g['lab_id'][j],g['stratum'][j],''.join('m' if v else '-' for v in np.isnan(g['coupon_features_raw'][j])),np.round([slope[j],r[j,-1],nonlinear[j],r[j,0],r[j,-2]],4))
for lab in np.unique(tr['lab_id']):
 print('ref lab',lab,[(len(a),round(np.median(a.mean(0)),3),round(np.median(a.std(0)),3)) for g in (tr,te,pi) for a in [g['ref_coupons'][g['lab_id']==lab]]]); print('feature nans', [(np.isnan(g['coupon_features_raw'][g['lab_id']==lab]).sum(0)).tolist() for g in (tr,te,pi)])
