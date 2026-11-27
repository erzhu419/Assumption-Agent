exec(open('/tmp/check_batch04.py').read().split("out=errors")[0])
o=s.solve_alpha_m96_w02_batch_04(d)
te=d['test']; e=o['predicted_strain']-g['te_clean']; f=o['initial_rate_pred']-g['te_true_rate0']
print('times',d['eval_times']);print('test fields',list(te));print('gt fields',list(g));
for i in np.argsort(np.max(abs(e),axis=1))[-12:]:
 print(i,'lab',te['lab_id'][i],'raw',np.round(te['coupon_features_raw'][i],3),'baseline',round(te['strain_0'][i],3),'error',np.round(e[i],4),'rate error',round(f[i],4))
for group in ('train','test'):
 a=d[group]; print(group,[(k,np.shape(a[k])) for k in a]); print('baseline quantiles',np.quantile(a['strain_0'],[0,.01,.1,.5,.9,.99,1]))
