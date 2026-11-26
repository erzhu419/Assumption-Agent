exec(open('/tmp/creep_baseline.py').read().split('for l in range(4):')[0])
case=pickle.load(open('/holdout_task/public_alpha_w03_batch_05.pkl','rb'))[0]; g=case['gt']; ztest=d['test']; ans=m.solve_alpha_w03_batch_04(d)
for k in ('taus','theta','vp_theta','stress_gains','stress_biases'): print('public reference',k,g[k])
for label,key in [('curve','predicted_strain'),('rate','initial_rate_pred')]:
 e=ans[key]-g['te_clean' if label=='curve' else 'te_true_rate0'];
 for l in range(4):
  for miss in (0,1):
   q=(ztest['lab_id']==l)&(np.isnan(ztest['coupon_features_raw']).any(1)==miss)
   if q.any(): print('public',label,l,miss,q.sum(),np.sqrt(np.mean(e[q]**2)),np.max(np.abs(e[q])))
print('test observed rate noise',np.quantile(ztest['instant_rate_obs']-g['te_true_rate0'],[0,.1,.5,.9,1]))
tau,b=m._fit(x[complete],h[complete],p[complete],dt,t,sy,y[complete],z[complete]); c,a=m._design(tau,x,h,p,dt,t,sy); E=y-c@b; rate=z-a@b
print('fit',tau,b)
for j in np.flatnonzero(np.max(abs(E),axis=1)>.2)[:25]: print(j,labs[j],strata[j],bool(complete[j]),round(h[j],4),round(rate[j],4),E[j].round(3))
for k in range(len(t)):
 e=abs(E[complete,k]); print('horizon',t[k],np.quantile(e,[.5,.9,.95,.99]),np.sum(e>.1))
