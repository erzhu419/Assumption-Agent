import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=m._bridge(d['calibration_load_cells']); x,xt=m._features(tr,te); _,xd=m._features(tr,di); p,h=m._loads(tr,b); pd,hd=m._loads(di,b); out=m.solve_batch(d); tau,beta=out['estimated_taus'],out['hereditary_weights']; print('FIT',tau,beta,'bridge',b,'times',t,'dt',dt)
for name,g,xx,pp,hh in [('train',tr,x,p,h),('diag',di,xd,pd,hd)]:
 a,c=m._design(tau,xx,pp,hh,dt,t); y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; e=y-a@beta; er=r-c@beta
 print(name,'curve abs by time',np.round(np.median(abs(e),axis=0),7),'rate abs quantiles',np.quantile(abs(er),[0,.5,.9,1]),'curve q',np.quantile(abs(e),[.5,.9,.99])); print('curve residual singular values',np.linalg.svd(np.clip(e,-np.quantile(abs(e),.95),np.quantile(abs(e),.95)),compute_uv=False))
 print('hold range',np.quantile(hh,[0,.5,1]),'past range',np.quantile(pp,[0,.5,1]))
