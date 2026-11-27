exec(open('/tmp/chamber_sparse.py').read().split('for name,F in models.items():')[0]); np.set_printoptions(precision=5,suppress=True,linewidth=170)
for mode in ['fixed','free']:
 yh,c,qp=fit(models['two affine amplitudes'],ids[~miss],mode); e=yh-y
 print(mode,qp,c,'bias by time',np.mean(e[~miss],axis=0)); print('rms complete/missing',metric(e[~miss]),metric(e[miss])); print('coupon mean early, late, rate and missing flag',np.c_[e[:,:7].mean(1),e[:,-7:-1].mean(1),e[:,-1],miss.astype(int)])
for label,g,X in [('train',tr,xt),('test',te,xe),('diag',d,x)]:
 print(label,'feature missing',np.isnan(g['coupon_features_raw']).sum(axis=0),'drift range',np.percentile(b[2]+X@b[3:],[0,10,50,90,100]),'time range',t[[0,-1]])
 if label=='train':
  A,R=a['_design'](g,X,s,dt,t,q); e=A@b-(tr['strain_curves_obs']-tr['strain_0'][:,None]); print('train rate by missing',[(bool(m),np.median(abs((R@b-tr['instant_rate_obs'])[np.isnan(tr['coupon_features_raw']).any(1)==m]))) for m in [0,1]]); print('train curve medians by time',np.median(e,axis=0))
