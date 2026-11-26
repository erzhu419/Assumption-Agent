import importlib.util, pickle, numpy as np, time
f='/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec=importlib.util.spec_from_file_location('candidate',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for p in pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb')):
 d,g=p['dataset'],p['gt']; start=time.perf_counter(); q=m.solve_viscoplastic_hereditary_revision(d); print('seconds',time.perf_counter()-start)
 e=np.abs(q['predicted_strain']-g['te_clean']); a=d['test']; ov=np.abs(a['stress_hold_0'])>d['yield_stress']; work=np.any(np.abs(a['past_stress'])>d['yield_stress'],axis=1)
 print('tau',q['estimated_taus'],'relative',abs(q['estimated_taus']/g['taus']-1),'vp',q['viscoplastic_weights'],'relative L2',np.linalg.norm(q['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']))
 print('MAE',e.mean(),{(int(i),int(j)):(int(k.sum()),float(e[k].mean())) for i in [False,True] for j in [False,True] if (k:=(ov==i)&(work==j)).any()}); print('sub',e[~(ov|work)].mean(),'above',e[ov|work].mean()); print('lab/stratum MAE',[(i,j,round(e[k].mean(),5)) for i in range(4) for j in range(3) if (k:=(a['lab_id']==i)&(a['stratum']==j)).any()]); print('clean rate noise RMS',np.std(a['instant_rate_obs']-g['te_true_rate0']))
