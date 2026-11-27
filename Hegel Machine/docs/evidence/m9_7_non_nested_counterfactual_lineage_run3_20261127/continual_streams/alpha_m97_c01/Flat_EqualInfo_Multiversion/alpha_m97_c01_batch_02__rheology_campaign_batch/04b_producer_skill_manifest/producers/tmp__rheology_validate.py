import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=item['dataset']; g=item['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); dt,t,past,hold,x=m._prepare(d)
a=m.solve_alpha_m97_c01_batch_02(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,'pred',a[k],'reference',g[j])
for k,j in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=a[k]-g[j]; print(k,'rmse by lab',[(int(c),round(np.sqrt(np.mean(e[te['lab_id']==c]**2)),6)) for c in np.unique(te['lab_id'])]); print('abs q50/q95/max',np.quantile(abs(e),[.5,.95,1]))
print('test stress max error',np.max(abs(past[n:]-g['te_hist_true'])),np.max(abs(hold[n:]-g['te_hold_true'])))
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; cp,rp,*_=m._basis(np.log(a['estimated_taus']),dt,t,past[:n],hold[:n],x[:n]); residual=yc-cp@a['hereditary_weights']; idx=np.unravel_index(np.argmax(abs(residual)),residual.shape); i,k=idx
print('largest residual',idx,'lab',tr['lab_id'][i],'raw feature',tr['coupon_features_raw'][i],'aligned',x[i],'observed curve delta',yc[i],'fitted curve delta',cp[i]@a['hereditary_weights'],'rate obs/fitted',yr[i],rp[i]@a['hereditary_weights'])
print('curve abs residual quantiles by time',np.quantile(abs(residual),[.5,.95,1],axis=0).round(4)); print('max feature abs by column',np.max(abs(x),axis=0))
for lab in np.unique(tr['lab_id']):
 q=tr['lab_id']==lab; p=m._fit(dt,t,past[:n][~q],hold[:n][~q],x[:n][~q],yc[~q],yr[~q]); c,r,*_=m._basis(p[:2],dt,t,past[:n][q],hold[:n][q],x[:n][q]); e1=c@p[2:]-yc[q]; e2=r@p[2:]-yr[q]; print('LOLO',lab,'taus',np.exp(p[:2]),'curve median/p95',np.quantile(abs(e1),[.5,.95]),'rate median/p95',np.quantile(abs(e2),[.5,.95]))
