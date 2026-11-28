exec(open('/tmp/memory.py').read().split('for label, x in')[0])
x=m['_features'](c,c)[0]; r=np.c_[np.ones(n),x]@w[2:]; z=np.c_[hold[:,None]*w[:2],np.zeros(n)]; E=Y-(z+r[:,None]*np.array([.26,1.8,.03]))@T.T
print('chamber row residual: lab / missing count / drift / projected corrections / joint RMS')
print(np.round(np.c_[lab,np.isnan(c['coupon_features_raw']).sum(1),r,E@np.linalg.pinv(T).T,np.sqrt((E**2).mean(1))],5))
print('deployment-only drift bounds',np.quantile(np.c_[np.ones(len(te['lab_id'])),m['_features'](te,te)[0]]@w[2:],[0,.1,.5,.9,1])); print('verification drift bounds',np.quantile(r,[0,.1,.5,.9,1]))
print('raw observations first/last/rate residual medians',np.median(E,axis=0))
