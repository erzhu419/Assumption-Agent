import glob,importlib.util,pickle,numpy as np
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=b['dataset']; g=b['campaign_diagnostics']; c=d['calibration_load_cells']; y=g['strain_curves_obs']; r=g['instant_rate_obs']; al=d['train']; te=d['test'];
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 s=importlib.util.spec_from_file_location('archive',f); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
 try:
  p=m.solve_batch(dict(d,test=g)); e=p['predicted_strain']-y; q=p['initial_rate_pred']-r
  print(f.split('/')[-1], 'taus',p['estimated_taus'].round(4),'weights',p['hereditary_weights'].round(4),'curve/rate MAD',np.round([np.median(abs(e)),np.median(abs(q))],5),'curve time MAE',np.median(abs(e),axis=0).round(4),'worst lab',max(np.median(abs(e[g['lab_id']==k])) for k in np.unique(g['lab_id'])),flush=True)
 except Exception as ex: print(f,repr(ex),flush=True)
print('cal range', c['true_channel_loads'].min(0),c['true_channel_loads'].max(0)); print('raw ranges train/test/diag',[(z['past_channel_raw'].min(),z['past_channel_raw'].max(),z['hold_channel_raw'].min(),z['hold_channel_raw'].max()) for z in (al,te,g)])
