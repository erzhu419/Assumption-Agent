exec(open('/tmp/chamber_models.py').read().split("summarize('benchtop'")[0])
features['m']=features['m0']+features['m1']
features['c0']=D(drift**3)*(tau[0]*g[0]-time)
features['c1']=D(drift**3)*(tau[1]*g[1]-time)
features['rat05']=H(drift**3/(1+.5*drift**2))
features['rat1']=H(drift**3/(1+drift**2))
features['m5']=D(past[:,-1])*(b[:2]@g)
for name, keys in [('cubic', ['f0','f1','steady','m0','m1','d3']),('shared memory cubic',['f0','f1','steady','m','d3']),('shared memory quadratic',['f0','f1','steady','m','d2']),('shared memory rational .5',['f0','f1','steady','m','rat05']),('shared memory rational 1',['f0','f1','steady','m','rat1']),('latest segment cubic',['f0','f1','steady','m5','d3']),('cubic in both transients',['f0','f1','steady','m','d3','c0','c1'])]:
 w,e=summarize(name,keys)
 print('  max lab overall/endpoint, no missing / missing rate MAE',np.round([max(abs(e[m,:-1]).mean() for m in labm),max(abs(e[m,-2]).mean() for m in labm),*[abs(e[m,-1]).mean() for m in [np.isfinite(q['coupon_features_raw']).all(1),~np.isfinite(q['coupon_features_raw']).all(1)]]],4))
