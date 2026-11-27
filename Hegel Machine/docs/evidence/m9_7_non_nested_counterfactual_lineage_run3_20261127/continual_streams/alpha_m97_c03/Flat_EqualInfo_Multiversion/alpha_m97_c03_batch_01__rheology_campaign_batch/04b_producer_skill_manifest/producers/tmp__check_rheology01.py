import importlib.util, pickle, numpy as np
base='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',base+'alpha_m97_c03_batch_01_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open(base+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; n=len(d['train']['lab_id'])
p,h,x,a,act,off,dt,t=m._prepare(d); args=(p,h,x,a,act); yy=np.c_[d['train']['strain_curves_obs']-off[:n,None],d['train']['instant_rate_obs']]
print('active',act[:n].sum(),act[n:].sum(),'stress errors',np.max(np.abs(p[n:]-g['te_hist_true'])),np.max(np.abs(h[n:]-g['te_hold_true'])))
for key in ('taus','theta','vp_theta'): print('reference',key,g[key])
for mode in ('gated','all','linear'):
 tau,w=m._fit((*[v[:n] for v in args],dt,t),yy.ravel(),mode)
 z=m._basis(np.log(tau),*args,dt,t,mode)[0]@w
 err=z[:n]-yy; ee=z[n:,:-1]+off[n:,None]-g['te_clean']; er=z[n:,-1]-g['te_true_rate0']
 print(mode,'tau',tau,'w',w.round(6),'train median abs',np.median(abs(err[:,:-1])),np.median(abs(err[:,-1])),'p95',np.quantile(abs(err),.95),'test rmse',np.sqrt(np.mean(ee**2)),np.sqrt(np.mean(er**2)))
 for b in (True,False):
  mask=act[n:]==b
  if mask.any(): print(' active',b,'max errors',abs(ee[mask]).max(),abs(er[mask]).max())
