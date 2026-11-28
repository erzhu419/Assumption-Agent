import importlib.util, pickle, numpy as np
p=importlib.util.spec_from_file_location('batch','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; keys=['a_panel','abi_panel','a_interact','raw_add','ab_add','bi_add','ab_trig','joint','plain']; summaries={}
for j in [np.arange(56,72),np.arange(16),np.arange(28,44),np.argsort(np.linalg.norm(x,axis=1))[-16:],np.random.default_rng(76).choice(72,16,replace=False)]:
 i=np.setdiff1d(np.arange(len(x)),j); f={**s,'train':{**t,**{k:t[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}
 ys,zs=m._predictions(f,keys,keys); r=m.solve_batch(f); ys['ensemble']=r['y_hat']; zs['ensemble']=r['state_pred']
 for k in ys:
  y=abs(ys[k]-t['y'][j]); z=abs(zs[k]-t['traj'][j]); summaries.setdefault(k,[]).append([y.mean(),z.mean(),z[:,-1].mean()]);
for k,a in summaries.items(): print(k,np.round(np.array(a).T,5), 'worst',np.round(np.max(a,0),5))
