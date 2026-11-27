import importlib.util,pickle,numpy as np,copy

def load(path):
 sp=importlib.util.spec_from_file_location('solver',path); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=load('/holdout_task/alpha_m96_w02_batch_02_solver.py'); new=load('/tmp/polymer_sparse.py')
e,=pickle.load(open('/holdout_task/public_alpha_m96_w02_batch_02.pkl','rb')); d,g=e['dataset'],e['gt']; c=d['calibration_load_cells']; lab=c['lab_id']; l=2; keep=np.flatnonzero(lab==l); spot=keep[np.argmax(abs(c['cmd_stress'][keep]))]
for name, gain, bias, sparse, spike in [('public',1,0,False,0),('nominal singleton',1,0,True,0),('shift singleton',1.045,.022,True,0),('shift singleton plus EMI',.955,-.021,True,.50),('EMI singleton',1,0,True,.50),('shift no checks',1.045,.022,True,None)]:
 dd=copy.deepcopy(d); cc=dd['calibration_load_cells'];
 for v in (dd['train'],dd['test']):
  mask=v['lab_id']==l
  for key in ['past_stress_cmd','stress_hold_cmd']: v[key][mask]=(v[key][mask]-bias)/gain
 cc['cmd_stress'][lab==l]=(cc['cmd_stress'][lab==l]-bias)/gain
 if sparse:
  mask=lab!=l
  if spike is not None: mask[spot]=True; cc['true_stress'][spot]+=spike
  dd['calibration_load_cells']={k:v[mask] for k,v in cc.items()}
 for sol in (base,new):
  out=sol.solve_alpha_m96_w02_batch_02(dd); err=out['predicted_strain']-g['te_clean']; cer=out['kkt_certificate']; z=dd['test']['lab_id']==l
  print(name,sol.__file__.split('/')[-1], 'tau_rel',round(np.max(abs(out['estimated_taus']/g['taus']-1)),5),'late_all',round(np.sqrt(np.mean(err[:,-1]**2)),5),'late_lab',round(np.sqrt(np.mean(err[z,-1]**2)),5),'w_err',round(np.max(abs(out['allocation_weights']-g['w_alloc_opt'])),6),'gap',format(cer['primal_objective']-cer['dual_lower_bound'],'.2e'))
