import runpy,pickle,numpy as np
m=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py');s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];ls=m['_ls'];aff=m['_affine'];np.set_printoptions(precision=4,suppress=True)
print('telemetry',s['telemetry_vec'],'summary',s['train']['summary'])
for key in ['panel_a','panel_b','panel_c']:
 d=s[key];u=d['u'];print(key,'u range',u.min(),u.max())
 for k,v in d.items():
  if k=='u':continue
  a=aff(u);w=ls(a,v);e=a@w-v
  print(k,'affine',w,'row residual quantiles',np.quantile(np.linalg.norm(e,axis=1),[0,.25,.5,.75,.9,1]))
  if key=='panel_c':
   p=m['_panel_operators'](d)[int(k[-1])-1];print('archived operator RMSE',np.mean((p(u)-v)**2)**.5)
  if key=='panel_a':
   b,w=m['_calibration'](d);print('robust coeff',b,w,'row residual quantiles',np.quantile(np.linalg.norm(u@w+b-v,axis=1),[0,.25,.5,.75,.9,1]))
