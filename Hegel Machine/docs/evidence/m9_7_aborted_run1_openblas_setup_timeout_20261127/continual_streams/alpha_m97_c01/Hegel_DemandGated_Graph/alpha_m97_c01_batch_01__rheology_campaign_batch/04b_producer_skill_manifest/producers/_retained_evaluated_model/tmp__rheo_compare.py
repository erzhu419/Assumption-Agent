import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c01_batch_01_solver.py'; sp=importlib.util.spec_from_file_location('cand',p); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0];d,g=it['dataset'],it['gt'];tr,te=d['train'],d['test'];data=m._prepare(d);n=len(tr['lab_id']);dt,t=d['past_dt'],d['eval_times'];xtr=tuple(a[:n] for a in data);xte=tuple(a[n:] for a in data);y=tr['strain_curves_obs']-tr['strain_0'][:,None];r=tr['instant_rate_obs']; truth=np.r_[g['taus'],g['theta'],g['vp_theta']];print('public reference parameters:',truth)
for mode in ('row','log','linear'):
 p=m._fit(xtr,dt,t,y,r,mode);a,b,*_=m._design(p[:2],xtr,dt,t,mode);u=m._dot(a,p[2:])-y;v=m._dot(b,p[2:])-r
 print(mode,'parameters',np.r_[np.exp(p[:2]),p[2:]])
 for active in (0,1):
  sel=(xtr[-1]>0)==active; print('training',active,'abs residual q50,q90,q99, max',np.quantile(abs(u[sel]),[.5,.9,.99,1]),np.quantile(abs(v[sel]),[.5,.9,.99,1]))
 a,b,*_=m._design(p[:2],xte,dt,t,mode);u=m._dot(a,p[2:])+te['strain_0'][:,None]-g['te_clean'];v=m._dot(b,p[2:])-g['te_true_rate0']
 print('test RMSE strain/rate',np.sqrt(np.mean(u*u)),np.sqrt(np.mean(v*v)))
 q=np.r_[np.log(g['taus']),truth[2:]];a,b,*_=m._design(q[:2],xte,dt,t,mode);print('reference-parameter reconstruction max',np.max(abs(m._dot(a,q[2:])+te['strain_0'][:,None]-g['te_clean'])),np.max(abs(m._dot(b,q[2:])-g['te_true_rate0'])))
