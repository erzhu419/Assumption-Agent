exec(open('/tmp/alpha_w02_batch04_cv.py').read().split('err =')[0].split('for a in s._archive_checks')[0])
y = np.column_stack((tr['strain_curves_obs']-tr['strain_0'][:,None], tr['instant_rate_obs']))
st = s._stress(tr,cal); p = s._plastic(st,dt,d['yield_stress']); x = s._features(tr,d['test'])[:len(y)]
tau, coef = s._fit(st,x,p,dt,time,y)
r = y-s._design(st,x,p,dt,time,tau)@coef
print('residual quantiles by time, final column onset rate')
print(np.round(np.quantile(r,[0,.01,.05,.25,.5,.75,.95,.99,1],axis=0),4))
print('curve median abs per row quantiles',np.quantile(np.median(abs(r[:,:-1]),axis=1),[0,.25,.5,.75,.9,.95,1]))
print('curve max abs per row quantiles',np.quantile(np.max(abs(r[:,:-1]),axis=1),[0,.25,.5,.75,.9,.95,1]))
for lab in range(4):
    m=tr['lab_id']==lab
    print('lab',lab,'mean',np.round(r[m].mean(0),4),'median',np.round(np.median(r[m],axis=0),4), 'count large', (abs(r[m])>.2).sum(0))
print('curve residual lag correlation',np.corrcoef(r[:,:-1].T)[0].round(3)); print('worst rows',np.round(r[np.argsort(np.max(abs(r),axis=1))[-10:]],4))
print('public truth comparison for baseline only:')
out=s.solve_alpha_w02_batch_04(d)
for k,g in zip(out,('taus','theta','vp_theta','te_clean','te_true_rate0')):
    pred,tgt=np.asarray(out[k]),np.asarray(case['gt'][g]); print(k,pred.shape,tgt.shape,'RMSE',np.sqrt(np.mean((pred-tgt)**2)),'max error',np.max(abs(pred-tgt)))
