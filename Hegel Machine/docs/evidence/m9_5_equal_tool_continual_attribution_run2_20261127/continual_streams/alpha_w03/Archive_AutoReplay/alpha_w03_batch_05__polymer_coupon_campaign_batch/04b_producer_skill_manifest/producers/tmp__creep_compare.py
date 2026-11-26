import importlib.util,pickle,numpy as np
paths=['/tmp/hegel_continual_archive/alpha_w03_batch_04_solver.py','/holdout_task/alpha_w03_batch_05_solver.py']
a=pickle.load(open('/holdout_task/public_alpha_w03_batch_05.pkl','rb'))[0]; d,g=a['dataset'],a['gt']; dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]; tr,te=d['train'],d['test']; n=len(tr['lab_id']); complete=np.isfinite(tr['coupon_features_raw']).all(1); labs=tr['lab_id']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; z=tr['instant_rate_obs']
rmse=lambda e: np.sqrt(np.mean(e**2))
for j,path in enumerate(paths,4):
 s=importlib.util.spec_from_file_location('creep',path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 ans=getattr(m,f'solve_alpha_w03_batch_0{j}')(d); x,h,p=m._prep(d)
 print('candidate',j,'tau',ans['estimated_taus'],'coef',ans['hereditary_weights'],'vp',ans['viscoplastic_weights'])
 for l in range(4):
  i=(labs!=l)&complete; v=labs==l; tau,b=m._fit(x[:n][i],h[:n][i],p[:n][i],dt,t,sy,y[i],z[i]); c,r=m._design(tau,x[:n][v],h[:n][v],p[:n][v],dt,t,sy); ec=c@b-y[v]; er=r@b-z[v]; ec[~complete[v]]-=er[~complete[v],None]*t
  good=np.abs(ec)<.2; test=te['lab_id']==l; test_complete=np.isfinite(te['coupon_features_raw']).all(1); cp=ans['predicted_strain']-g['te_clean']; rp=ans['initial_rate_pred']-g['te_true_rate0'];
  print('lab',l,'OOF curve inliers',round(rmse(ec[good]),6),'OOF complete rate',round(rmse(er[complete[v]]),6),'public curve/rate complete and missing',[(round(rmse(cp[q]),6),round(rmse(rp[q]),6)) for q in (test&test_complete,test&~test_complete) if q.any()])
